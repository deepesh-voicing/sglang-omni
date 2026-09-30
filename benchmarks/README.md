# SGLang Omni Benchmarks

Benchmark suite for Voicing-TTS on SGLang Omni, covering performance (latency,
throughput, RTF, streaming TTFC), accuracy (WER), speech quality (UTMOS), and
speaker similarity.

## Directory Structure

```
benchmarks/
├── tasks/          # Task logic: tts (speech send layer, SeedTTS stages), asr (WER client)
├── metrics/        # Metric computation (performance, WER, UTMOS, speaker similarity)
├── dataset/        # SeedTTS loader + download helpers
├── benchmarker/    # Framework: runner, data structures, utilities
├── eval/           # Entry-point scripts
├── tts_serving/    # TTS serving harness and Docker contract
├── cache/          # (gitignored) dataset caches
└── results/        # (gitignored) evaluation outputs
```

## WER Needs an External ASR Server

This repository serves only Voicing-TTS; it does not serve ASR models. WER is
computed by transcribing the generated audio through an external
OpenAI-compatible `/v1/audio/transcriptions` server that you start separately
and pass with `--asr-port` (and `--asr-host` when it is not on `127.0.0.1`).
`--asr-model-path` is the model name sent in each transcription request
(default `Qwen/Qwen3-ASR-1.7B`); names containing `whisper` are sent in 30 s
chunks. `--generate-only`, `--utmos-only`, and `--similarity-only` do not need
an ASR server.

## Quick Start

```bash
# 0. Prepare dataset (once)
python -m benchmarks.dataset.prepare --dataset seedtts

# 1. Convert a checkpoint (once)
python -m sglang_omni.models.voicing_tts.convert_checkpoint \
    Qwen/Qwen3-TTS-12Hz-1.7B-Base checkpoints/voicing-tts-12hz-1.7b-base

# 2a. Full pipeline: start Voicing-TTS, generate, stop it, then score WER on the
#     external ASR server listening on port 8001
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model checkpoints/voicing-tts-12hz-1.7b-base \
    --server-config examples/configs/voicing_tts_1_7b.yaml \
    --port 8000 --asr-port 8001 \
    --output-dir results/voicing_tts_en --lang en --max-samples 50 --concurrency 8

# 2b. Generate only against a running server (speed metrics, no transcription)
sgl-omni serve --config examples/configs/voicing_tts_1_7b.yaml --port 8000
python -m benchmarks.eval.benchmark_tts_seedtts \
    --generate-only --use-existing-server --stream \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model checkpoints/voicing-tts-12hz-1.7b-base --port 8000 \
    --output-dir results/voicing_tts_en --max-samples 50 --concurrency 8

# 2c. Transcribe only (reuses audio from a prior generate run; no TTS server)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model checkpoints/voicing-tts-12hz-1.7b-base \
    --output-dir results/voicing_tts_en --lang en --asr-port 8001

# 2d. CustomVoice checkpoint: built-in speaker, no voice cloning
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model checkpoints/voicing-tts-12hz-1.7b-customvoice \
    --server-config examples/configs/voicing_tts_1_7b_customvoice.yaml \
    --no-ref-audio --task-type CustomVoice --voice Vivian \
    --port 8000 --asr-port 8001 \
    --output-dir results/voicing_tts_customvoice_en --lang en --max-samples 50

# 2e. VoiceDesign checkpoint: voice described by instructions
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model checkpoints/voicing-tts-12hz-1.7b-voicedesign \
    --server-config examples/configs/voicing_tts_1_7b_voicedesign.yaml \
    --no-ref-audio --task-type VoiceDesign \
    --instructions "A warm, natural young adult voice." \
    --port 8000 --asr-port 8001 \
    --output-dir results/voicing_tts_voicedesign_en --lang en --max-samples 50

# 3a. Offline UTMOS (naturalness MOS prediction) scoring on existing output
python -m benchmarks.eval.benchmark_tts_seedtts \
    --utmos-only --output-dir results/voicing_tts_en --device cuda:0

# 3b. Offline speaker similarity (voice resemblance) scoring on existing output
python -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --output-dir results/voicing_tts_en --device cuda:0
```

## Eval Scripts

| Script | Task | API |
|--------|------|-----|
| `eval/benchmark_tts_seedtts.py` | Voicing-TTS speed + WER + UTMOS + speaker similarity on SeedTTS | `/v1/audio/speech` (+ external `/v1/audio/transcriptions` for WER) |
| `eval/benchmark_tts_serving.py` | TTS serving contract and stress | `/v1/audio/speech`, raw PCM streaming, `/v1/audio/speech/batch`, WebSocket, `/v1/audio/voices` |
| `eval/bench_sweep.py` | Open-loop request-rate sweep over N same-GPU replicas | `/v1/audio/speech` via `benchmark_tts_seedtts.py` |

See [tts_serving/README.md](tts_serving/README.md) for the TTS serving
benchmark design, harness contract, scenario matrix, and Docker usage.

`benchmark_tts_seedtts.py` is a two-phase pipeline: phase 1 generates and
persists WAVs while the TTS server runs, and phase 2 transcribes them through
the external ASR server. Use `--generate-only` or `--transcribe-only` to run a
single phase. Without `--use-existing-server`, phase 1 starts and stops a
managed Voicing-TTS server from `--model` (and `--server-config` when set) and
forwards `--max-running-requests`, `--max-queued-requests`,
`--cuda-graph-max-bs`, and `--quantization` to its `tts_engine` stage.
`--concurrency` and `--max-concurrency` are equivalent.

Voice-cloning references come from the SeedTTS meta file: the default
`--ref-format flat` sends `ref_audio`/`ref_text`, while `--ref-format
references` sends `references=[{audio_path, text}]`. `--no-ref-text` keeps the
reference audio but drops its transcript. Reference audio on this endpoint is a
filesystem path, so it is not client-encoded inside the request timer; if the
server restricts local media with `--allowed-local-media-path`, that directory
must cover the dataset cache. `--subtalker-dosample-ratio` interleaves sampled
and greedy `subtalker_dosample` requests deterministically by sample index.

`--seed`, `--temperature`, `--top-p`, `--top-k`, and `--repetition-penalty` are
recorded in the speed results. Seeded sampling is not free: SGLang's seeded
sampler hashes every vocabulary entry per token, so use the same seed setting in
both arms of an A/B comparison and never compare seeded against unseeded
absolute numbers. `--fingerprint` records the client environment and the
server `/v1/models` identity.

Warmup runs in the benchmark client after the server is available. By default
it repeats one sample `--warmup` times concurrently (defaulting to the
concurrency); `--warmup 0` disables it. Repeating one sample avoids warming
per-sample server caches, such as the reference-audio cache, for the measured
requests.

`--concurrencies 1,16 --repeats 5 --generate-only` sweeps concurrency levels.
One repeat keeps the directory `c<level>`; further repeats write
`c<level>_r<repeat>`. Every row in `concurrency_sweep.json` aggregates each
level's repeats (mean, min, max, n per metric), with `per_repeat` holding each
raw summary. Tail percentiles need at least 100 measured samples; below that
p99 interpolates the two slowest requests and the benchmark logs a warning.

`--sustained-overshoot --max-queued-requests N --generate-only` holds
open-loop arrivals above `max_running_requests + max_queued_requests` for
`--overshoot-duration-s` and writes `<output-dir>/overshoot/sustained_overshoot.json`.

## TTS Quality Evaluation

To evaluate the overall quality and vocal resemblance of synthesized speech, the benchmark suite supports offline evaluation using UTMOS (naturalness MOS prediction) and Speaker Similarity (vocal fidelity). Running `benchmark_tts_seedtts.py` with `--utmos-only` or `--similarity-only` loads the respective pre-trained predictor and scores the previously generated audio in the output directory without requiring the TTS or ASR servers to be running.

### UTMOS (Naturalness)

UTMOS (UTokyo-Saru Lab MOS Prediction) is a Mean Opinion Score (MOS) predictor model used to evaluate the naturalness and overall quality of synthesized speech. SGLang Omni provides an offline UTMOS evaluator backed by the `balacoon/utmos` JIT model on Hugging Face.

- **Model Weights Cache**: By default, weights (`utmos.jit`) are downloaded from Hugging Face on the first run and cached at `~/.cache/sglang-omni/utmos` (override via `UTMOS_CACHE_DIR` environment variable).
- **Warm the Cache (Optional)**:
  ```bash
  python -m benchmarks.metrics.utmos --warm-cache
  ```
- **Outputs (`utmos_results.json`)**:
  - `summary`: section for summary metrics
    - `utmos_mean`: Mean predicted MOS score in `[1, 5]` (higher is better).
    - `utmos_median`: Median predicted MOS score.
    - `utmos_p5` / `utmos_p95`: 5th and 95th percentile scores to identify worst-case outliers or top-performing samples.
    - `total_samples`: total number of samples in the dataset.
    - `evaluated`: number of samples that were evaluated.
    - `skipped`: number of samples that were skipped.
  - `config`: evaluation configuration parameters.
  - `per_sample`: list of individual score results for each evaluated sample.

### Speaker Similarity (Vocal Fidelity)

Speaker Similarity evaluates how closely the voice of synthesized speech matches the reference prompt audio.

- **Model Weights Cache**: By default, model weights (`wavlm_large.pt` and `wavlm_large_finetune.pth`) are downloaded from Hugging Face and cached at `~/.cache/sglang-omni/speaker_sim` (override via `SEEDTTS_SIM_CACHE_DIR` environment variable).
- **Warm the Cache (Optional)**:
  ```bash
  python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
  ```
- **Outputs (`similarity_results.json`)**:
  - `summary`: section for summary metrics
    - `speaker_similarity_mean`: Mean cosine similarity score scaled by 100.0 (higher is better).
    - `total_samples`: total number of samples in the dataset.
    - `evaluated`: number of samples that were evaluated.
    - `skipped`: number of samples that were skipped.
  - `config`: evaluation configuration parameters.
  - `per_sample`: list of individual score results for each evaluated sample.

## Adding a New Task

Add an eval script under `eval/` that reuses the task helpers in `tasks/tts.py`
(`make_tts_send_fn`, `run_seedtts_transcribe`, `run_seedtts_utmos`,
`run_seedtts_similarity`) and the ASR client in `tasks/asr.py`.

## Datasets

Download helpers live in `benchmarks/dataset/prepare.py`:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts       # full SeedTTS
python -m benchmarks.dataset.prepare --dataset seedtts-mini  # smoke-test subset
python -m benchmarks.dataset.prepare --dataset seedtts-50    # 50-sample subset
```

All datasets are pre-warmed into the default HuggingFace cache via
`datasets.load_dataset(repo_id)`. SeedTTS Arrow repos stage audio to
process-local tempfiles at load time; no manual `--local-dir` step is needed.
