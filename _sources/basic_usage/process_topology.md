# Process Topology, Replicas, and GPU Sharing

Four mechanisms decide how an SGLang-Omni pipeline occupies its GPUs. They
compose, and each answers a different question:

1. **Process topology** defines which stages run together.
2. **Process replicas** define how many copies run.
3. **Placement** defines where they run.
4. **CUDA MPS** improves kernel scheduling between processes on one GPU.

| Mechanism | Configured per | What it changes | What it does not do |
|---|---|---|---|
| Process topology | `StageConfig.process` | non-TP logical Process membership; TP materializes per rank | replica count, placement |
| Process replica | `PipelineConfig.processes[<name>]` | copies a whole logical Process, sticky per request | model parallelism for one request, MPS |
| Placement | `replica_devices` | the GPU each replica or rank lands on | GPU context scheduling |
| CUDA MPS | `mps` | kernel overlap between colocated CUDA contexts | routing, weight or KV sharing |

Which one you want:

| Your question | Mechanism | What it will not solve |
|---|---|---|
| Which stages share one OS process and its local state? | Process topology | replica count and GPU placement |
| Which bottleneck Process needs more capacity? | Process replica | model parallelism for a single request |
| Which GPU does each replica run on? | `replica_devices` | CUDA context scheduling |
| How do colocated processes use idle compute? | CUDA MPS | replica creation and request routing |

## Stages and process topology

A stage is one logical execution unit of the pipeline DAG. It declares its
factory, wiring (`next`, `stream_to`, `wait_for`), GPU, TP size, runtime
resources, and process name.

For a non-TP stage, `StageConfig.process` defines Process membership. Stages
sharing a Process Name share an OS process, Python heap, asyncio event loop,
and local dispatch path; GPU members also share one CUDA context. A TP stage
owns its logical Process and materializes one OS process per rank.

A logical Process is a grouping, spawn, and placement boundary. It is not a
recovery boundary: any child stage process dying still stops the pipeline.

The Voicing-TTS pipeline runs `preprocessing`, `tts_engine`, and `vocoder` in
one Process named `pipeline` by default. To move the vocoder into its own
Process:

```yaml
stages:
  tts_engine:
    process: pipeline
  vocoder:
    process: vocoder
```

Inspect the resolved result with `sgl-omni config resolve --config <config.yaml>
--show config`.

## Process replicas

A replica copies a whole logical Process, not a single stage. Stages inside one
Process are copied together under the same replica index. The runtime names
physical instances `<name>@rN`, while models and routes keep referring to
logical names.

At admission the coordinator picks one replica per replicated Process the
request touches. That binding rides the message and stays fixed for the whole
request lifetime, across payload, stream, completion, and abort paths. The
default policy is per-Process thread-safe round robin, chosen independently for
each Process.

```yaml
processes:
  pipeline:
    num_replicas: 2
    replica_devices: [1, 2]
```

## Placement

`replica_devices` sets the GPU each replica uses, including replica 0, and
overrides the placement of every GPU stage inside that replica. CPU stages are
unaffected and stay on the host.

A GPU Process with `num_replicas > 1` must declare `replica_devices`: `N`
device ids for a non-TP Process, `N x T` for a Process with TP size `T`. Every
GPU stage factory declares `device: str | None = None` and
`gpu_id: int | None = None` and resolves them through
`sglang_omni.utils.device.resolve_concrete_device`, so any GPU stage can be
replicated; a factory that lacks `gpu_id` is refused at startup by name, and
`tests/unit_test/test_stage_device_contract.py` keeps every model on that
contract. Configs never set `factory.gpu_id`; a `factory.device` names a
device type only (for example `cpu` to keep a stage on the host) and must not
carry an index — the card comes from `stage.gpu` or `replica_devices`.
Different replicas may repeat a device id, which is how same-GPU data
parallelism is expressed:

```yaml
processes:
  pipeline:
    num_replicas: 2
    replica_devices: [1, 1]
```

That places two non-TP replicas on GPU 1. It does not mean two ranks of one TP
replica may share a GPU.

When `replica_devices` colocates Process groups on one GPU, every GPU stage
involved must declare `gpu_memory_fraction`, whether or not the general
colocation check is enabled. The value is a placement-time budget, not a
runtime memory limit:

```yaml
stages:
  vocoder:
    gpu_memory_fraction: 0.10
```

## CUDA MPS

MPS only schedules multiple CUDA contexts on one GPU. It does not create
replicas, choose routes, or share weights, KV, or CUDA graphs. When a single
context already saturates the GPU, MPS can add contention and tail latency
rather than throughput.

The runtime manages the daemon itself. Modes (`--mps` on the CLI or `mps:` in
the config, default `off`):

* `off`: MPS is never touched.
* `auto`: enabled on every GPU hosting two or more single-GPU, non-TP CUDA
  processes of this pipeline, including process replicas.
* `on`: one eligible process is enough, and an MPS-incapable platform is a hard
  error instead of a warning.

```bash
sgl-omni serve --config <config.yaml> --mps auto --port 8091
```

The daemon is shared per physical GPU, keyed by device UUID. See
[Same-GPU Data Parallelism with CUDA MPS](mps_dp.md) for the full lifecycle,
verification, and operator notes.

## Putting it together

Configure in this order:

1. decide which stages share a process;
2. choose the replica count;
3. place each replica on a GPU;
4. enable MPS when processes share a GPU.

### Replicas on one GPU

Repeat the device id and declare the memory budget for every GPU stage on
that card. Without `--mps`, the replicas time-slice the GPU. The
`preprocessing` stage declares no GPU of its own, so it needs no fraction. The
fractions below are illustrative; size them for your card:

```yaml
config_cls: VoicingTTSPipelineConfig
name: voicing-tts-dp2
model_path: checkpoints/voicing-tts-12hz-1.7b-base

stages:
  tts_engine:
    gpu: 0
    gpu_memory_fraction: 0.35
    engine:
      mem_fraction_static: 0.30
      max_total_tokens: 30000
  vocoder:
    gpu: 0
    gpu_memory_fraction: 0.10

processes:
  pipeline:
    num_replicas: 2
    replica_devices: [0, 0]
```

### Replicas on one GPU with MPS

Same config, plus the MPS flag. Both replicas are single-GPU, non-TP processes
on GPU 0, so `auto` enables MPS there:

```bash
sgl-omni serve --config <config.yaml> --mps auto --port 8091
```

## Performance and correctness

Replicas mainly help under queueing and higher concurrency. At low concurrency
they can be slightly slower, because the parallelism gain does not cover
routing and process overhead. MPS helps only when colocated processes have
overlappable GPU work.

Validate a topology change on: serial output consistency, concurrent routing
and request isolation, abort and recovery, clean exit of processes, ports, and
GPU memory, and, when enabled, MPS attachment.

## Migration

Removed interfaces and their replacements are covered in
[Process Topology Migration](process_topology_migration.md): express process
membership with `StageConfig.process`, and replica count and placement with the
top-level `processes` block.
