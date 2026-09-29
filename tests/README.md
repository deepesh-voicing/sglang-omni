## Folder Structure
```text
tests/
├── README.md
├── __init__.py
├── utils/
│   ├── __init__.py
│   └── ci_cpu_contention.py
├── test_ci/
│   ├── test_mps_native.py
│   └── test_omni_missing_dependencies.py
├── test_model/
│   ├── conftest.py
│   ├── omni_router_utils.py
│   ├── rust_router_config.py
│   ├── tts_ci_config.py
│   ├── test_tts_ci.py
│   ├── test_tts_consistency_artifacts.py
│   ├── test_tts_serving_ci.py
│   └── test_voicing_tts_batch_invariance.py
└── unit_test/
    ├── benchmarks/
    ├── ci/
    ├── cli/
    ├── client/
    ├── config/
    ├── cpu/
    ├── diagnostics/
    ├── fixtures/
    ├── model_runner/
    ├── models/
    ├── mps/
    ├── npu/
    ├── pipeline/
    ├── platforms/
    ├── preprocessing/
    ├── profiler/
    ├── quantization/
    ├── relay/
    ├── router/
    ├── sampling/
    ├── scheduling/
    ├── scripts/
    ├── serve/
    ├── utils/
    ├── vendor/
    ├── voicing_tts/
    └── xpu/
```

## How To Add A Test


General rules:

- Protect user-visible contracts and component ownership, not incidental implementation structure.
- Keep imports thin and consistent. If a test monkeypatches a module object,
  call through that module alias instead of mixing direct symbol imports.
- Reuse existing helpers and fakes before adding another scheduler, relay, or
  lifecycle helper.
- Add a one-sentence docstring to non-obvious contract tests.
- Do not add root-level `tests/test_*.py` files.


## Markers

Markers are registered in `pyproject.toml` under `[tool.pytest.ini_options]`.
Apply markers for resource requirements and CI selection, and use them to
filter runs.

- `benchmark`: GPU performance, parity, and deployment tests, primarily in
  `test_model/`. They may require converted Voicing-TTS checkpoints and
  substantial GPU memory; per-test docstrings state their hardware
  requirements.
- `tts_stage(name)`: in-file CI stage selector for TTS benchmarks.
  Combined with `--tts-stage` (see `test_model/conftest.py`).
- `accelerator`: tests that require accelerator hardware. Pair this marker with
  a backend-specific availability guard so the test skips cleanly when that
  hardware is unavailable. The marker must be declared unconditionally; a
  runtime or artifact-based skip does not assign the test to the accelerator
  CI job. Marker filtering happens after test modules are imported, so keep
  accelerator runtime probes such as `torch.cuda.is_available()` and
  `torch.cuda.device_count()` out of module scope and collection-time skip
  conditions. Perform them in the test body or a fixture instead.


## Root Files

- `README.md`: This file. It explains test ownership and where new tests belong.
- `__init__.py`: Keeps `tests` importable as a package.
- `utils/`: Shared helpers used by model CI tests: server lifecycle, metric
  checks, the external WER ASR fixture, and the CPU contention sampler.

## Checkpoints

The model tests serve converted Voicing-TTS checkpoints. Convert a stock
checkpoint once with:

```bash
python -m sglang_omni.models.voicing_tts.convert_checkpoint \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base checkpoints/voicing-tts-12hz-1.7b-base
```

`VOICING_TTS_MODEL_PATH` and `VOICING_TTS_CUSTOM_VOICE_MODEL_PATH` point the
TTS CI presets at the Base and CustomVoice checkpoints; they default to
`checkpoints/voicing-tts-12hz-1.7b-{base,customvoice}`.

## `test_model/`

End-to-end and model CI tests. These are allowed to depend on real servers,
model checkpoints, benchmark artifacts, optional packages, and GPU/runtime
resources.

Expected command (GPU benchmark subset):

```bash
pytest tests/test_model -m benchmark -v -s
```

Relevant model CI ownership:

- `test_tts_ci.py`: default TTS CI gate. It starts the TTS managed router
  with two one-GPU workers using the selected preset, runs the full SeedTTS EN
  set (1088 samples) in non-streaming / streaming stages at concurrency 16,
  and frees the server GPUs before the WER, speaker-similarity, and UTMOS
  checks. WER transcribes the saved WAVs on an external OpenAI-compatible ASR
  server at concurrency 4, since this repository serves Voicing-TTS only.
- `tests/utils/__init__.py` owns the `external_wer_asr` fixture. Set
  `WER_ASR_PORT` (and `WER_ASR_HOST` if the server is not local) to run the WER
  stages; `WER_ASR_MODEL` names the served ASR model and defaults to
  `Qwen/Qwen3-ASR-1.7B`. Without `WER_ASR_PORT` the WER stages skip. GitHub CI
  reads all three from repository variables of the same name.
- `test_tts_consistency_artifacts.py`: CPU-only stage-3 check that compares
  TTS non-stream and streaming `speed_results.json` under
  `${OMNI_CI_HOME}/tts-stage-results/{nonstream,stream}/`.
- `test_tts_serving_ci.py`: production-mimic serving benchmark through a
  two-worker router, driven by the spec in `benchmarks/tts_serving/`. Its mixed
  reference values are unset until they are recalibrated for Voicing-TTS, so
  the stage reports numbers without gating on them.
- `test_voicing_tts_batch_invariance.py`: deterministic inference checks that
  compare batch-size-1 and batch-size-8 audio for the Base and CustomVoice
  checkpoints. `VOICING_TTS_TEST_MODEL`,
  `VOICING_TTS_TEST_CUSTOM_VOICE_MODEL`, `VOICING_TTS_TEST_CUSTOM_VOICE`,
  `VOICING_TTS_TEST_CUSTOM_LANGUAGE`, and `VOICING_TTS_TEST_DATASET` override
  the defaults.
- `conftest.py`: TTS CI options and the session-wide `OMNI_CI_CPUSET` pinning
  with its contention report.
- CLI flags `--tts-stage {tts-stage-1-nonstream,tts-stage-2-stream,tts-stage-3-consistency,all}`
  and `--concurrency {1,2,4,8,16,all}`: scope a TTS CI sweep without
  editing source.
- CLI flag `--tts-ci-model {voicing-tts,voicing-tts-custom-voice}`:
  select the TTS CI model preset for `test_tts_ci.py` without editing source.
  Defaults to the `TTS_CI_MODEL` environment variable, then `voicing-tts`.
  The CustomVoice preset gates nothing until it is calibrated on the CI host.
- CI env alignment on the H100 repro host: `source .github/scripts/ci_env.sh`
  then `source omni/bin/activate`.
  Omni CI (`omni-ci.yaml`) runs PR Test (`test.yaml` unit tests) and then TTS
  CI after one shared setup, which also converts the Voicing-TTS checkpoints.
  A PR Test failure does not skip TTS CI; only a failed setup blocks the chain.
  Full WER sweep: `.github/scripts/run_all_wer_ci_aligned.sh` (milestones on
  stdout; details in `/tmp/wer_ci_tts.log`).
- GPU handoff between stages: `.github/scripts/delete_gpu_process.sh --kill-orphans` (kills orphan
  spawn/router workers, waits for VRAM below threshold).

## `test_ci/`

- `test_mps_native.py`: native MPS lifecycle smoke on one GPU. It serves the
  CustomVoice checkpoint with the vocoder in its own process, so `--mps auto`
  sees two clients. Not wired into a workflow; run it manually on a GPU host.
- `test_omni_missing_dependencies.py`: the requirement probe the CI venv
  setup uses to decide whether to reinstall.

## `unit_test/`

Fast contract tests that should run without model downloads or real server
startup. Keep these focused on the smallest component that owns the behavior.
Most unit tests run on CPU. Accelerator-dependent cases use the `accelerator`
marker and explicit backend availability guards.

Expected command:

```bash
pytest tests/unit_test -q
```

Select CPU cases with `-m "not accelerator"`. Run hardware cases with
`-m accelerator` on a compatible accelerator; check the reported skips
to confirm the intended hardware paths actually ran.

Choose the location by the behavior contract being protected, not by the file
that happened to contain an older version of the test.

- `unit_test/voicing_tts/`: Voicing-TTS unit tests:
  - pipeline config and registry contracts
  - the Transformers compatibility patches applied before the vendored
    library is imported
  - OmniScheduler-backed AR stage factory wiring
  - request mapping for `ref_audio` / `ref_text` and `references`
  - incremental codec-to-vocoder ordering, priority batching, fallback parity,
    CUDA stream handoff, and abort/failure cleanup
  - streaming vocoder decode slots: per-thread pinned staging with one reused
    completion event, bad rows raised from `resolve()`, pageable fallback after
    pinned allocation failure, and slot retirement or process-lifetime
    retention when a launch or event wait fails (with a real CUDA reuse case)
  - model-owned default preservation for language and sampling parameters
  - Base, CustomVoice, and VoiceDesign request validation
  - voice-clone reference validation
  - pipeline payload state serialization
  - code-predictor CUDA-graph bit-identity, capture-failure fallback, top-k
    ladder masking, and enablement gating (env, `disable_cuda_graph`, TP)
  - reference and speaker encoder CUDA graphs, NPU sampling, and prefill
    retraction.
- `unit_test/pipeline/`: Model-agnostic pipeline tests:
  - compile
  - placement planning
  - runtime wiring
  - runtime schema/adapter behavior
  - coordinator behavior
  - process replicas: whole-process stage expansion, instance naming, device
    assignment, process-level binding, and logical-to-physical routing
  - stage routing
  - centralized comm router selection, data-reference serialization, ack
    lifecycle, and sender backpressure release
  - stage process environment
  - relay handling
  - GPU memory accounting helpers
  - IPC lifecycle
  - scheduler batching
  - stream termination drains the runner before publishing the terminal
    output and preserves the finish reason (`test_scheduler.py`).
  - scheduler errors
  - scheduler concurrency
  - async-decode drop-stale handling, including per-token field reslicing on
    decode and extend/mixed batches
  - scheduler callable contracts, including sync wrappers and callable objects
    that return awaitables.
- `unit_test/relay/`: Low-level data-plane relay tests:
  - shared-memory relay byte movement, cleanup, and handle lifecycle on CPU
  - CUDA-IPC relay metadata/open/close behavior for GPU tensor handoff; CUDA
    tests require CUDA and multi-GPU coverage is hardware-gated
  - these tests prove transport mechanics, not full pipeline throughput,
    NVLink selection, or production backpressure behavior; keep those covered
    in `unit_test/pipeline/` integration tests and GPU benchmarks.
- `unit_test/benchmarks/`: TTS SeedTTS benchmark configuration and phase
  ordering (including releasing the TTS server before the external ASR phase),
  dataset loading regressions, playback continuity, sweeps, and runtime
  resource-monitoring, PID-scoping, aggregation, and provenance coverage.
- `unit_test/utils/`: Shared utility tests:
  - pinned CUDA staging primitives (`cuda_staging`): exact-size growth,
    reusable events, non-blocking completion queries, device checks, and
    record/query/synchronize failure handling. Failed records invalidate
    completion reads until a later record succeeds. CPU tests use stand-ins;
    `accelerator` cases cover in-flight D2H queries and cross-device use.
  - checkpoint and device helpers, and the shared SnakeBeta activation.
- `unit_test/model_runner/`: Shared model-runner contract tests:
  - arch override pool sizing: the talker engine's KV pool takes the talker's
    layer count through SGLang's layer resolver (the Voicing-TTS talker at 28
    layers under a 36-layer root config).
  - Voicing-TTS talker registration with the SGLang model registry.
  - prefill CUDA Graph usage: isolated counter state, replay/eager phase
    classification, executed-bucket counts, and JSON-safe model-info output.
  - sampling seeds, penalty history, repetition penalty, rollout log-probs,
    and weight checks.
- `unit_test/models/`: Model registry contract tests:
  - static TTS `ModelCapabilities` declarations, registry lookup, aliases, and
    launcher startup logging.
- `unit_test/scheduling/`: Shared scheduling-service unit tests:
  - early-tail flushing remains opt-in when stream completion arrives
    before the final payload (`test_streaming_vocoder.py`).
  - `EvictHeapRadixCache` eviction-order equivalence against upstream
    `RadixCache` on randomized traces, heap boundedness and recovery after a
    full drain, and reset-then-reuse behavior.
  - deferred request admission completion, abort, and dependency-failure
    semantics.
  - breakable prefill CUDA Graph policy: backend/cap/bucket validation, shared
    cap-derived ladders, disable precedence, and capability/attestation wiring.
  - `ReferenceEncodeService` cache, same-key single-flight, timeout, failure,
    and revalidation semantics.
  - `StageOutputCache` thread safety: concurrent get/put byte-accounting,
    non-negative capacity validation, identity-checked removal that preserves
    newer replacements,
    the `remove_if` eviction predicate evaluated outside the lock (re-entrant
    and deadlock-free), and concurrent remove_if/put state integrity.
  - `build_sglang_server_args` on a real mini checkpoint: the record leaves the
    builder resolved once, and the CUDA Graph config it declared reads back
    through `resolution_result` and the generation batch policy accessors.
- `unit_test/router/`: SGLang-Omni Router unit tests:
  - router CLI/config behavior
  - worker metadata and health-state contracts
  - request routing, proxying, and streaming relay
  - worker selection policy behavior
  - managed launcher command construction and cleanup.
- `unit_test/serve/`: In-process serving API unit tests:
  - generation-stage SGLang server-args role mapping and CLI override capability boundaries
  - OpenAI-compatible speech, voices, models, health, and admin routes
  - speech request validation, error mapping, batching, speaker caching, and
    the speech WebSocket
  - streaming response framing and failure semantics.
- `unit_test/profiler/`: Request-level profiler unit tests:
  - `RequestEvent` schema and JSONL emit/append behavior
  - concurrent emit safety under multiple threads
  - lifecycle (start / stop / run_id mismatch / stage substitution)
  - timeline reconstruction, stage breakdown, hop breakdown, malformed-line tolerance.
- `unit_test/quantization/`: Tests for the compatibility layer on top of
  SGLang's native quantization (`sglang_omni/quantization.py`):
  - `resolve_quant_config` discovery from root/nested sub-configs and
    `compression_config`, plus edge cases (missing/empty quantization_config)
  - FP8 detection (with/without weight_block_size), weight_scale_inv reciprocal
    conversion, and error handling (empty/zero/non-finite/non-float scale tensors)
  - `get_weight_preprocessor` contract: identity by default (native block-FP8),
    FP8 reciprocal preprocessor only when `fp8_scale_inverted=True`, and
    nested config traversal.
- `unit_test/config/`: Pipeline config schema, resolver, CLI overrides, config
  paths, and dotted set/patch behavior.
- `unit_test/ci/`: CPU unit tests for CI configuration and infrastructure,
  including CPU allocation and contention accounting, TTS model selection and
  preset contracts, slash-command labels, and installed-pin verification.
- `unit_test/mps/`: MPS configuration, decision, device, daemon control,
  manager, runtime, spawn-environment, and runner-wiring contracts, plus the
  helpers `test_ci/test_mps_native.py` uses. These use synthetic state and do
  not launch an MPS workload.
- `unit_test/cli/`: SGLang backend registration, option forwarding,
  configuration-only launch behavior, and backend-specific help ownership.
- `unit_test/client/`: Audio conversion, resampling, channel preservation, and
  speech response decoding.
- `unit_test/diagnostics/`: GPU inventory and dependency diagnostics using
  mocked NVML and subprocess probes; no physical GPU is required.
- `unit_test/preprocessing/`: Reference-audio cache identity and
  `test_resource_connector.py`, which covers the
  `MultiModalResourceConnector` local-media policy: bare local paths and
  `file://` URLs are both scoped to `allowed_local_media_path` once it is
  configured, and `..` traversal and symlink escapes are rejected before
  MediaIO is called.
- `unit_test/sampling/`: Random, explicit, and deterministically derived
  per-row sampling-seed contracts.
- `unit_test/vendor/`: Server-argument publication boundaries and the vendor
  RMSNorm patch staying on the fused-op dispatch path (dtype fallback,
  zero-token contract, kwargs passthrough).
- `unit_test/platforms/`, `unit_test/cpu/`, `unit_test/npu/`, `unit_test/xpu/`:
  platform detection, device selection and placement contracts, and installer
  scripts. No accelerator is required.
- `unit_test/scripts/`: The repository lint scripts (`check_if_else.py` and
  `check_leading_underscore.py`).
- `unit_test/fixtures/`: Shared fakes, plus the runtime accelerator probe
  (`accelerator.py`, `require_cuda(min_devices)`) that `accelerator`-marked
  tests call in the test body. `mini_checkpoint.py` writes a two-layer Llama
  `config.json` for tests that need a record the SGLang resolution pipeline can
  resolve end to end. Single-test helpers should stay local until a second
  test needs them.
