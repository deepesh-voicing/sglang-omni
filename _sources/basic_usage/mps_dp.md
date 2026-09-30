# Same-GPU Data Parallelism with CUDA MPS

> TL;DR: When a tuned Voicing-TTS replica still leaves the GPU idle, running several complete replicas on the same GPU under CUDA MPS can increase per-GPU throughput.

A common data-parallel deployment assigns one GPU to each replica. When a tuned replica still leaves substantial GPU headroom, colocating multiple replicas on the same GPU can improve per-GPU throughput.

Same-GPU data parallelism runs several complete serving replicas on one GPU and lets [CUDA MPS](https://docs.nvidia.com/deploy/mps/index.html) share the GPU between them. This is a conditional and ongoing optimization. We are excited to share it and call for the community to join the exploration.

![Multiple host chains plus CUDA MPS filling the idle GPU](../_static/image/same-gpu-dp-mps.svg)

## Native runtime support (`--mps`)

The runtime can manage MPS itself for the processes of **one pipeline**. When a
pipeline colocates two or more single-GPU stage processes on one GPU (for
example a frontend process next to the generation process, or process-level
replicas, see [Process Topology, Replicas, and GPU Sharing](process_topology.md)),
pass `--mps auto`:

```bash
sgl-omni serve --config <config.yaml> --mps auto
```

Modes (`--mps` on the CLI or `mps:` in the pipeline config; default `off`):

* `off`: MPS is never touched.
* `auto`: MPS is enabled on every GPU that hosts two or more single-GPU,
  non-TP CUDA processes of this pipeline. GPUs with one process and TP groups
  run without MPS.
* `on`: a single eligible process is enough, and an MPS-incapable platform is
  a hard error instead of a warning. Use `on` for same-GPU data parallelism:
  every `serve --mps on` on one GPU joins the same daemon.

Both `auto` and `on` reject startup before acquiring MPS state when one process's
resolved placement spans more than one physical GPU; use `mps=off` for that
placement. Factory CUDA devices use the narrowed worker's local namespace:
`cuda:0` is compatible with any single-GPU placement, while a nonzero `cuda:N`
is excluded because narrowing a worker to one UUID makes its only valid CUDA
ordinal `cuda:0`.
Pipeline-edge transport remains the responsibility of the existing router and
relay layers; it does not participate in MPS eligibility.

The daemon is shared per physical GPU (keyed by device UUID): MPS merges
kernels only for clients of one server, so the first serve creates the
daemon, later serves join it, and the last one to leave drains the clients
and quits it. Logical GPU ordinals are resolved once against the parent
process's CUDA visibility and then grouped by physical UUID; `auto` counts the
combined client processes in each physical group. Pipeline or stage
environment defaults must not override `CUDA_VISIBLE_DEVICES` or
`CUDA_DEVICE_ORDER` while native MPS is enabled; set them on the parent command
instead, or use `mps=off`. Same-GPU DP is therefore just N serve commands:

```bash
sgl-omni serve --config examples/configs/voicing_tts_1_7b.yaml --mps on --mem-fraction-static 0.35 --port 8807
sgl-omni serve --config examples/configs/voicing_tts_1_7b.yaml --mps on --mem-fraction-static 0.35 --port 8808
```

Start replicas one after another and give each an explicit memory budget
(`--mem-fraction-static` or the stage-qualified
`--tts_engine.engine.max_total_tokens`), for the same KV-sizing
reasons described under the script recipe below. Route traffic with the
[Omni Router](omni_router.md).

Process-level replicas can size their pools in bytes instead, with
`engine.kv_cache_bytes` per stage and `total_reserve_bytes` for the replica's
full footprint. The budgets are then checked against the card before any
replica is spawned: colocation requires a declared footprint, the summed
reserves and fractions must fit the card, and the summed KV pools alone must
too. `engine.kv_cache_bytes` and `engine.max_total_tokens` are mutually
exclusive on one stage, since the lower token cap would silently shrink the
byte-derived pool.

The runtime owns the full lifecycle. Every managed process is verified against
the daemon's client list before serving starts, because a process that misses
the pipe directory silently falls back to time slicing. A watchdog fails the
pipeline if daemon identity or control access is lost mid-serving. Shutdown
re-evaluates the current client list, drains this serve's clients, and quits the
daemon only when no other serve still owns it.

If a managed worker does not exit before the shutdown timeout, the runtime
terminates that directly owned child process and reaps it before the launcher
exits, even when that directly owned worker is also an MPS client. It sends no
additional signal based on an MPS snapshot or client PID, and never
automatically signals the daemon, an unknown descendant, or a GPU-wide process
set. Process ownership and shared MPS state are handled independently: if
daemon identity, client ownership, or control state cannot be proved after the
workers are gone, the owner file is marked `retained`, its lock is released,
and the state directory is preserved. The current command then exits with a
detailed non-zero error instead of keeping a CLI owner alive.

Dirty state is never repaired automatically. A join requires the native
`nvidia-cuda-mps-control.pid` identity, a responsive control socket, and every
published owner lease to still be held. After a hard kill (SIGKILL, OOM kill,
node crash), even an idle daemon or one dead co-owner makes the next start
preserve the state and fail with owner/client details and safe cleanup guidance.
An unlocked or retained owner blocks every later start until an operator has
inspected and cleaned the state. Existing healthy co-owners keep serving, but
new owners cannot join and no process retries cleanup automatically. Clean up
and start again. A normal shutdown leaves nothing behind.

Operator notes: state lives under `/tmp/sglang-omni-mps-<user>/<gpu-uuid>/`
(`SGLANG_OMNI_MPS_STATE_ROOT` overrides it). Serves that are meant to share
one GPU must use the same state root, or they cannot discover each other's
daemon and will run separate MPS servers that time-slice against each other.
The state root is created with mode `0700`; an existing root must already be a
non-symlink directory owned by the current user with that mode. Native MPS
rejects `CUDA_MPS_PIPE_DIRECTORY` in the parent, pipeline, or stage environment
instead of overwriting or joining an external daemon.

## Deploy

The steps below are one continuous flow. We provide `examples/mps_dp/launch.sh` to manage the private MPS daemon and serving replicas for one run. It records replica processes, ports, and logs, starts replicas sequentially, verifies their KV capacity and MPS attachment, and tears down only the run it recorded. Detailed instructions are as follows:

1. **Choose the GPU and NUMA node.**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
export GPU_ID=0
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU_ID)
BUS=${BUS,,}; BUS=${BUS:4}
NODE=$(cat /sys/bus/pci/devices/$BUS/numa_node)   # if -1, set the node explicitly
numactl -H | grep "node $NODE cpus"
```

Pick a GPU that is idle, then find its NUMA node from the PCI bus id (drm card ordinals do not always match nvidia-smi ordinals). Choose non-overlapping physical CPU-core blocks from that node, one block per replica.

2. **Launch the replicas.**

Convert the checkpoint first (see the [Voicing-TTS cookbook](../cookbook/voicing_tts.md)), then launch three Voicing-TTS replicas:

```bash
CONFIG=examples/configs/voicing_tts_1_7b.yaml N=3 MAX_TOTAL_TOKENS=30000 CORE_BLOCKS="0-9 10-19 20-29" bash examples/mps_dp/launch.sh up
```

The pipeline config supplies the model and per-replica runtime settings. The launcher environment supplies host-local placement, including the GPU, replica count, and CPU blocks. The launcher resolves the GPU's NUMA node, assigns local ports automatically, starts a private MPS daemon, waits for each replica's health check before starting the next, and verifies MPS attachment. The CPU blocks above are an example; derive the correct non-overlapping blocks for your own CPU topology.

The Voicing-TTS pipeline defaults `mem_fraction_static` to `0.85`; `MF` can override it. Launching replicas sequentially avoids overlapping memory profiling and CUDA-graph capture during startup.

Identical `--mem-fraction-static` flags do **not** mean identical KV capacity. `--mem-fraction-static` budgets model weights and the KV pool against the GPU memory available when each replica starts. Roughly, the profiled KV memory is the requested fraction of free memory measured before model loading, minus model and fixed runtime allocations. It is a per-replica budget, not an additive share of the card. Because replicas start sequentially, earlier ones have already reserved memory, so later ones see a smaller free pool and allocate fewer KV tokens even when every flag is the same (in one run, three sequential `mf=0.27` replicas received 97,503 / 53,149 / 20,961 KV tokens).

Memory profiling does not coordinate KV allocation across independent replica processes. For `N > 1`, every replica must resolve the same KV capacity, which can come from either sizing knob but never both:

* `engine.kv_cache_bytes` in the pipeline config sizes every replica's pool deterministically; the launcher verifies all replicas resolved the same capacity and rejects a simultaneous `MAX_TOTAL_TOKENS`, since a lower token cap would silently shrink the byte-derived pool.
* Without a byte budget, the launcher requires one common `max_total_tokens` value, from the pipeline config or from `MAX_TOTAL_TOKENS`, and rejects startup unless every replica resolves exactly that capacity.

Either cap applies independently to each replica; it is not divided across the pool. It is also independent of the request-level `max_new_tokens` limit and does not distribute requests between replicas.

The Voicing-TTS example configs do not pin `max_total_tokens`, so the command above sets `MAX_TOTAL_TOKENS`. The `30000` value is an example, not a hardware default: leave GPU memory headroom for non-KV runtime allocations, including the colocated vocoder, and recalculate the cap after changing the model, GPU, runtime, replica count, memory settings, or CUDA-graph settings. If a replica cannot allocate the common cap, lower it or reduce the replica count.

A KV pool smaller than the worst-case demand of `max_running_requests` long requests makes `max_running_requests` an admission ceiling rather than a guarantee that that many requests can decode concurrently. If the pool fills, SGLang retracts requests and returns them to the waiting queue until KV capacity becomes available.

3. **Drive every replica to saturation.**

Use one dedicated client per replica and drive all replicas in parallel. The goal is to keep every replica saturated. With equivalent replicas, random or round-robin routing can distribute a shared ingress across the pool; fill-one-then-next is another possible strategy. Equal KV capacity makes the replicas comparable, but it does not by itself balance their queues. Validate the routing policy and per-replica saturation under your workload.

4. **Verify MPS attachment.**

MPS should be verified carefully. Four things are easy to conflate: environment variables set, daemon running, an MPS server exists, and the replica processes you launched are actually attached as clients. Only the last makes the comparison valid, and a replica that missed the pipe directory falls back to time-slicing without any error. The launcher verifies every replica against the MPS client list, writes the server-to-client PID mapping to `mps_attach.txt`, and fails startup if any replica is not attached.

5. **Route traffic.**

For easy deployment, you can register each replica endpoint with the [Python Router](python_router.md). Keep the router's `--max-connections` at least as large as the total offered concurrency. Router scheduling policies have not been benchmarked for a colocated pool, so confirm that the selected policy keeps every replica driven and meets your workload's latency and throughput requirements.

6. **Tear down safely.**

Stop new traffic, then run the teardown command printed by the launcher:

```bash
bash examples/mps_dp/launch.sh down <RUN_ID>
```

On a shared host, only touch processes you launched, and never treat "the GPU is empty" as the success condition. The launcher stops only the replica processes recorded for the selected run, waits for their MPS clients to detach, and then stops the private MPS daemon. It keeps the run state whenever cleanup cannot be confirmed.

## Common questions

**Throughput has plateaued, so why is the GPU still idle?**

Serving throughput depends on more than the GPU's peak compute. It also depends on how much parallel work a single replica exposes per step, how fast the host side handles scheduling and stage handoffs, and the request-length and batching distribution. A single TTS replica can have a full request queue and still leave most of the GPU's SMs idle; adding a second independent replica can improve GPU idle and throughput together. So one process's serving path is not keeping the card fed, but the cause is not a single CPU function: multiple host execution paths, batching behavior, and a latency-bounded decode shape can all contribute.

**Replicating the weights costs VRAM. What does that buy?**

Same-GPU DP does not save VRAM; it spends more of it. It copies the weights per replica and gives each replica its own, smaller KV pool. What it buys is the otherwise idle compute, reclaimed. That trade pays off only when a tuned single replica leaves the GPU idle (so there are idle SMs to fill) and the model is small enough that its weights are a modest slice of the card, so two or three full replicas still fit. On a compute-bound model, or one too large to hold several weight copies, extra replicas buy little.

**Why does this pay for TTS models and not for general LLM serving?**

Memory fit is the enabling condition, not the cause. The cause is idle that a single engine cannot reclaim, and TTS-style AR audio models produce it on two axes at once:

- *Latency-capped batch shapes.* Streaming first-chunk latency pins the per-replica batch small, and a 0.6B or 1.7B AR model at that batch runs low-occupancy kernels. The usual LLM remedy — batch deeper in one engine — spends the latency budget the product is built around.
- *Host-heavy serving path.* Sampler pools, vocoder scheduling, chunk assembly, and HTTP streaming do per-step host work that rivals the GPU step time, so a single process idles the card temporally between launches. N processes overlap one replica's dispatch bubble with another's kernels; this is also why same-GPU DP scaling is sensitive to the CPU cores allotted per replica.

A large dense transformer inverts every part of this: its decode batch can grow until the GEMMs saturate the SM array (continuous batching in one engine already multiplexes requests over one weight copy), SM utilization is high at serving batch sizes so MPS has no idle to harvest — only contention to add — and at tens of GiB per weight copy, same-card replicas stop fitting at all. The scaling tools there are TP/PP/EP within one engine, not DP behind MPS. Rule of thumb: colocate replicas when a tuned single replica holds roughly ≤60% SM-active under its latency SLO and N× the footprint fits; otherwise scale the batch, not the process count.

## Measure your own setup

### Prepare the baseline

The single-replica baseline decides whether same-GPU DP is worth it, and an under-driven baseline makes DP look better than it is. Tune and measure one replica first, then treat its throughput, latency, and GPU utilization as the number every DP configuration has to beat.

* **Sweep concurrency to the plateau.** Raise client concurrency until throughput stops climbing, and read the scheduler log lines (`#running-req`, `#queue-req`) at each step rather than assuming a good operating point.
* **Know the admission limit.** Voicing-TTS serves with `max_running_requests=64` by default; raise it with `sgl-omni serve --tts_engine.engine.max_running_requests N --tts_engine.engine.cuda_graph_max_bs N` (the CUDA-graph capture range must cover the admission limit, and raising it costs capture memory). Whether the default cap binds depends on the runtime, so check the queue, do not assume.
* **Separate client from server.** Client concurrency is not the active generation batch: requests beyond the admission limit wait in the scheduler queue, and requests also spend time in the other pipeline stages.
* **Prerequisites.** NVIDIA CUDA MPS available with GPU compute mode `Default`, so a per-user daemon needs no root; enough GPU memory for every replica's common KV cap plus roughly fixed per-replica overhead (weights, vocoder, MPS context); non-overlapping CPU core blocks, one per replica, on the GPU's NUMA node (on SMT machines logical CPUs `N` and `N + ncores` are often the same physical core, so check `lscpu -e=CPU,CORE,NODE`); and enough offered concurrency to saturate each replica, not just the pool.

To check whether one tuned replica is below GPU saturation under your real workload before adopting DP:

```bash
nvidia-smi dmon -i $GPU_ID -s um -d 5                        # coarse utilization
nsys profile --gpu-metrics-devices $GPU_ID --gpu-metrics-set gh100 \
  -d 60 -o one_replica -f true sleep 63                      # device-level SM-active
```

Low SM activity at the tuned single replica's peak may indicate reclaimable headroom; confirm it with a controlled DP comparison before relying on it. If SM activity is already near the ceiling, stop here.

### Evaluate

Whether same-GPU DP helps is easy to measure incorrectly, so hold the comparison to the same discipline for every configuration:

| Control | Why it matters |
|---|---|
| tune the single replica to its throughput plateau | keeps the baseline from being artificially weak |
| hold total GPU and CPU resources fixed | separates replica splitting from simply adding resources |
| give each replica dedicated CPU cores | keeps replicas from contending for host dispatch |
| saturate each replica separately | keeps the DP pool from being under-fed |
| pin software and runtime settings | makes the comparison reproducible |
| report latency and unsuccessful runs | avoids showing only the best throughput |

## Limits and next steps

1. **Generality is not fully validated.** Gains depend on the model size, GPU, CPU allocation, and workload. We believe same-GPU DP is a promising direction for smaller models on GPUs with ample memory and compute headroom, but the experimental coverage is still incomplete.

2. **KV sizing is hardware- and workload-specific.** The launcher enforces equal per-replica KV capacity through a common `MAX_TOTAL_TOKENS`. A sizing procedure that generalizes across models, runtimes, and GPU configurations still requires further study.

3. **Router and scheduler still need a deeper dive.** Both the router and the SGLang Omni scheduler need further optimization. On the router side, better routing strategies for a colocated pool are clearly required. On the scheduler side, a more ambitious question is whether multiple replicas can share one large KV cache. That direction is extremely challenging, and we believe the potential payoff is correspondingly large.

Same-GPU DP with MPS can recover idle GPU time on host- or dispatch-bound serving today, but broader validation and the work above are still unfinished. If this direction interests you, or you have results from other GPUs or workloads that confirm or challenge these findings, we would like to work with you.
