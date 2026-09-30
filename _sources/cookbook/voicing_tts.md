# Voicing-TTS

Voicing-TTS is a discrete multi-codebook text-to-speech model served from
`sglang_omni/models/voicing_tts`. It performs fast voice cloning from a short reference clip,
supports 10 languages, and streams 24 kHz speech with low latency. The `12hz` in the checkpoint
names refers to the codec **frame rate** (12 acoustic frames per second), not the playback
sample rate. SGLang-Omni serves three variants — Base (voice cloning), CustomVoice (built-in
speakers), and VoiceDesign (a voice described in text) — at `0.6B` and `1.7B` through the same
`preprocessing → tts_engine → vocoder` pipeline and the OpenAI-compatible `/v1/audio/speech`
endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md). No extra
packages are required: the tokenizer, speech-tokenizer, and speaker-encoder code Voicing-TTS
needs is vendored under `sglang_omni/vendor/voicing_tts`.

## Convert a Checkpoint

Voicing-TTS checkpoints are converted from the public Qwen3-TTS 12Hz checkpoints. The converter
copies the checkpoint and renames its model types for the Voicing-TTS loaders. It accepts a local
checkpoint directory or a Hugging Face repo id (downloaded on demand, no token required):

```bash
python -m sglang_omni.models.voicing_tts.convert_checkpoint \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  checkpoints/voicing-tts-12hz-1.7b-base
```

The target directory must not exist yet. Convert each variant you plan to serve into the
directory its example config expects:

| Source checkpoint | Converted directory | Config |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `checkpoints/voicing-tts-12hz-0.6b-base` | `examples/configs/voicing_tts_0_6b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `checkpoints/voicing-tts-12hz-1.7b-base` | `examples/configs/voicing_tts_1_7b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `checkpoints/voicing-tts-12hz-0.6b-customvoice` | `examples/configs/voicing_tts_0_6b_customvoice.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `checkpoints/voicing-tts-12hz-1.7b-customvoice` | `examples/configs/voicing_tts_1_7b_customvoice.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `checkpoints/voicing-tts-12hz-1.7b-voicedesign` | `examples/configs/voicing_tts_1_7b_voicedesign.yaml` |

Each config sets `model_path` to its converted directory; pass `--model-path` to serve a
checkpoint stored elsewhere. Keep `voicing-tts` in a Base checkpoint's directory name and end
it with `base`: the server recognizes a Base checkpoint from that path and only then enables
uploaded reference voices (`/v1/audio/voices`) for named-voice cloning.

## Server Configuration

The pipeline is `preprocessing → tts_engine → vocoder`. First startup can take several minutes
while the `tts_engine` captures CUDA graphs.

```bash
# 0.6B
sgl-omni serve \
  --config examples/configs/voicing_tts_0_6b.yaml \
  --port 8000
```

```bash
# 1.7B
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b.yaml \
  --port 8000
```

### Ascend NPU baseline

The NPU configurations use SGLang's `ascend` attention backend for the Talker
and PyTorch SDPA for both Speech Tokenizer instances. Talker graph capture
(`cuda_graph`), private Talker compile, and asynchronous vocoder decode stay
disabled. The 0.6B Base
configuration has been validated with 16 concurrent requests; the other
configurations retain a conservative single-request baseline.

```bash
# 0.6B Base
sgl-omni serve \
  --config examples/configs/voicing_tts_0_6b_npu.yaml \
  --port 8000
```

Use `voicing_tts_1_7b_npu.yaml`, `voicing_tts_0_6b_customvoice_npu.yaml`, or
`voicing_tts_1_7b_voicedesign_npu.yaml` for the other supported checkpoints.
The 0.6B and 1.7B files intentionally have separate `mem_fraction_static`
starting values. Calibrate configurations that retain the single-request
baseline on the target NPU before increasing `max_running_requests` or any
vocoder batch limit.

### Deterministic Inference

Dynamic batching can change Voicing-TTS codec and waveform outputs even when the
prompt, reference audio, and seed are unchanged. Both the 0.6B and 1.7B Base
checkpoints provide an opt-in deterministic mode:

```yaml
enable_deterministic_inference: true
```

When enabled, the same prompt, reference audio, and seed produce byte-identical
PCM across runtime batch sizes. This mode reduces throughput because it
serializes reference preprocessing and vocoder decoding and disables both the
initial and follow-up vocoder graph-capture paths (`initial_cuda_graph` and
`followup_cuda_graph`), so it is disabled by default.

### Overload / admission policy

Two SGLang generation-stage knobs bound how the server behaves past saturation:

| Knob | Meaning | Voicing-TTS default |
|---|---|---|
| `--tts_engine.engine.max_running_requests` | Concurrent running slots | `64` |
| `--tts_engine.engine.max_queued_requests` | Waiting-queue depth before fast-reject | `64` |

Every request enters the waiting queue first, so `max_queued_requests`
must be **≥ 1**. Capacity is about `running + queued`. Extra arrivals get
HTTP **503** (`The request queue is full.`) before preprocessing, or later
if the AR waiting queue or request-build backlog is full. Voicing-TTS
defaults to 4 request-build workers with pending depth 16.

### Breakable prefill CUDA graphs

Every Voicing-TTS checkpoint (Base, CustomVoice, VoiceDesign) defaults to the
breakable prefill CUDA-graph backend with a token ladder up to 512:

| Knob | Meaning | Default |
|---|---|---|
| `--tts_engine.engine.cuda_graph_backend_prefill` | Prefill graph backend (`breakable` or `disabled`) | `breakable` |
| `--tts_engine.engine.cuda_graph_bs_prefill` | Prefill token-count ladder to capture | shared ladder through `512`, plus a `1` bucket |
| `--tts_engine.engine.cuda_graph_max_bs_prefill` | Cap for the ladder | top of the ladder |

The default is the shared ladder with one bucket added. A replay falls
back to eager when its bucket exceeds twice the real token count, and the
shared ladder starts at 4, so a 1-token prefill lands in bucket 4 and
misses. Measured over 3203 prefills at 10 and 20 RPS, 1301 of them (40.6%)
are exactly one token, and they are the only shapes that fall back: 2 and
3 already replay inside bucket 4. Adding the single `1` bucket takes the
fallback rate to zero.

Opt out with `--tts_engine.engine.cuda_graph_backend_prefill disabled`. The
default costs extra graph capture during startup. Raising
`cuda_graph_max_bs_prefill` on its own regrows the default ladder to the
new cap; declaring `cuda_graph_bs_prefill` yourself keeps your list as is.

To change the ceiling, set `max_running_requests` and `max_queued_requests`
together:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_0_6b.yaml \
  --tts_engine.engine.max_running_requests 32 \
  --tts_engine.engine.max_queued_requests 16 \
  --port 8000
```

Stepped `--concurrencies` is a closed-loop client: it never holds more than
N in-flight requests, so past-ceiling load is a burst that drains. Keep
offered load above `max_running_requests + max_queued_requests` for a
duration with open-loop sustained overshoot:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server --stream \
  --model checkpoints/voicing-tts-12hz-0.6b-base \
  --port 8000 \
  --max-running-requests 32 \
  --max-queued-requests 16 \
  --sustained-overshoot \
  --overshoot-duration-s 10 \
  --max-samples 64
```

Arrivals default to `2 × capacity` (`--request-rate` overrides). Stats are
on successes only; artifacts land in `<output-dir>/overshoot/`.

A closed-loop `--concurrencies 16,32,48,64` sweep is still available for
comparing healthy vs past-ceiling points, but it does not hold overshoot. Each
concurrency writes inspectable artifacts under `<output-dir>/c<N>/`.

### Prefill Admission Coalescing

Under concurrent load, the `tts_engine` stage can coalesce prefill admission:
instead of admitting each prepared request into its own prefill batch, the
scheduler can briefly hold admission so that multiple ready requests are
prefilled together.

A prefill step has a largely fixed scheduler cost, so fuller batches can reduce
prefill overhead. The end-to-end benefit depends on whether that saving
outweighs the extra admission delay and any resulting reduction in decode
occupancy.

Coalescing is **off by default** and opt-in through the `tts_engine` factory
configuration:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b.yaml \
  --tts_engine.factory.prefill_coalesce_requests 2 \
  --tts_engine.factory.prefill_coalesce_wait_ms 30 \
  --port 8000
```

or per-stage in YAML:

```yaml
stages:
  tts_engine:
    factory:
      prefill_coalesce_requests: 2
      prefill_coalesce_wait_ms: 30.0
```

The gate engages only when `prefill_coalesce_requests >= 2`. Once engaged,
admission is released as soon as any of the following holds:

- decode is idle, so a ready request can start immediately;
- the waiting queue reaches `prefill_coalesce_requests`;
- the oldest waiting request has waited `prefill_coalesce_wait_ms`.

`prefill_coalesce_wait_ms` is therefore an upper bound on the added admission
wait. Admission may be released earlier if the target queue size is reached.

The values above are an example for the Voicing-TTS workload and are not intended
as universal defaults. Match both `prefill_coalesce_requests` and
`prefill_coalesce_wait_ms` to the workload you actually serve. Coalescing is
most useful when natural prefill batches are small and a short hold can increase
batching without materially reducing decode occupancy. If the wait is too long,
the reduced decode occupancy can offset the prefill savings.

Leave coalescing disabled for latency-sensitive traffic or workloads where the
added wait does not produce enough additional batching.

### Process topology

By default all three stages share one process. Per-request reference
preprocessing (speech-tokenizer encode, speaker embedding, prompt embedding)
then competes with the AR scheduler and the vocoder for the same interpreter,
which caps single-replica throughput once concurrency passes ~32. Moving the
preprocessing stage to its own process removes that contention: the stage
loads a prompt-only frontend (embedding tables, text projection, predictor
codec embeddings, speaker encoder) plus the speech tokenizer, together about
2.2 GB of GPU memory for the extra process on a 1.7B checkpoint, and ships the
prepared prompt tensors to the engine through the payload. Every GPU stage must
then declare a memory fraction, and the engine's static fraction has to agree
with the one it declares:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b.yaml \
  --preprocessing.process tts_frontend \
  --preprocessing.gpu 0 \
  --preprocessing.gpu_memory_fraction 0.05 \
  --tts_engine.gpu_memory_fraction 0.75 \
  --tts_engine.engine.mem_fraction_static 0.75 \
  --vocoder.gpu_memory_fraction 0.12
```

`--vocoder.process vocoder` composes with it (lower `tts_engine` to 0.72 and
give the vocoder 0.15). Six SeedTTS samples at a fixed seed produced identical
PCM in both layouts; the extra process costs its CUDA context plus the frontend
weights.

## Synthesizing Speech

### Text-only Requests

Voicing-TTS Base checkpoints require a reference clip. Text-only requests are supported by CustomVoice and VoiceDesign checkpoints; see [TTS Model Usage](../basic_usage/tts.md) for those launch commands.

### Voice Cloning

The `references` field accepts `audio_path` (a local path or HTTP URL) and `text` (the
transcript of that clip). Supplying the transcript enables in-context-learning (ICL) mode and
materially improves cloning quality; omitting it falls back to speaker-embedding (x-vector)
mode.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "voice": "default",
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for `references[0].audio_path` and
`references[0].text`.

A Base checkpoint can also register a reference clip once through `/v1/audio/voices` and reuse
it by name in `voice`; see [Uploaded Voices](../basic_usage/tts.md#uploaded-voices). Base has
no built-in speakers, so a request without reference fields whose `voice` is neither `default`
nor an uploaded name returns HTTP 400.

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "voicing-tts",
        "voice": "default",
        "input": "Get the trust fund to the bank early.",
        "references": [{
            "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            "text": "We asked over twenty different people, and they all said it was his.",
        }],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

Non-streaming responses include `X-Finish-Reason: stop` after codec EOS or
`X-Finish-Reason: length` when generation reaches `max_new_tokens`. A `length`
response still contains decodable audio, but the utterance may be incomplete.
Batch responses expose the same value as each item's `finish_reason`.

#### Leading silence in x-vector mode

Without a reference transcript the talker starts cold, and on most requests it
opens with several frames of room-tone silence before speaking, which the
listener hears as extra time to first audio. The engine therefore excludes
silence codec ids from the first two frames of x-vector-only requests. The
silence ids are derived at startup from the checkpoint's own codec (it encodes
stationary noise up to -50 dBFS) and logged once. ICL, CustomVoice, and
VoiceDesign requests are not affected.

On SeedTTS EN (1088 clips, no reference text, seed 0):

| Checkpoint | Frames masked | Median onset | Onsets > 160 ms | WER | Speaker similarity |
|---|---|---|---|---|---|
| 1.7B Base | 0 | 495 ms | 95% | 0.854% | 61.00 |
| 1.7B Base | 2 (default) | 155 ms | 47% | 0.904% | 60.87 |
| 0.6B Base | 0 | 480 ms | 86% | 1.532% | 58.41 |
| 0.6B Base | 2 (default) | 75 ms | 23% | 1.072% | 58.10 |

The mask changes which codec ids the first frames can sample, so outputs differ
from the unmasked model. To turn it off:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b.yaml \
  --tts_engine.factory.leading_silence_mask_frames 0 \
  --port 8000
```

Details and the full sweep are in
[the leading-silence benchmark](../benchmarks/voicing_tts_leading_silence.md).

### Language Hint

`language` biases the model toward a target language. It defaults to `auto` (let the model
detect). Supported languages are Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, and Italian.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "voice": "default",
    "input": "今天天气不错，就该出去晒晒太阳。",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "language": "Chinese"
  }' \
  --output output.wav
```

### Streaming

Set `"stream": true` and `"response_format": "pcm"` to receive raw PCM audio
chunks in real time:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

Streaming returns `audio/pcm` 16-bit mono PCM bytes with sample-rate metadata in
the response headers. See [TTS Model Usage](../basic_usage/tts.md#use-python) for a
full Python raw PCM consumer.

All three task types (Base/reference-cloning, CustomVoice and VoiceDesign) use
true incremental codec and vocoder streaming, for both this HTTP endpoint and
`/v1/audio/speech/stream` WebSocket sessions with `stream_audio=true`. Pass
`"stream_codec_output": false` on a request, or launch with
`--preprocessing.factory.stream_codec_output false`, to restore whole-utterance
decoding.

Streamed CustomVoice output on validated voice/language pairs (currently
Ryan/English with default sampling) withholds the model's silent bootstrap
codec frame, removing about 80 ms of leading silence from the first chunk. The
frame still feeds the vocoder, so every later sample is unchanged, and a
runtime silence check emits the audio unmodified whenever the first frame is
not actually silent. Opt out per request with
`"suppress_bootstrap_silence": false` or per deployment with
`--vocoder.factory.suppress_bootstrap_silence false`.

When `initial_codec_chunk_frames` is omitted, Voicing-TTS ramps its first chunks
`1 -> 2 -> 4` codec frames before the steady stride, so first audio leaves after a
single AR step while the playback cushion is rebuilt within four chunks. Pass an
explicit value to trade continuity against time-to-first-audio.
Utterances that finish in fewer than the first chunk's generated codec frames never reach the
first chunk, so their audio arrives complete in a single final flush.

#### Codec decoding defaults

Streaming decodes run on the stateful incremental codec by default: each
follow-up chunk decodes only its fresh frames against per-stream state held in
a preallocated arena, steady-state cohorts replay CUDA graphs whose decode step
is `torch.compile`d, and the follow-up workers collect for 4 ms. Startup spends
about a minute compiling the steady shapes. The left-context decoder remains
available as a rollback:

```yaml
stages:
  vocoder:
    factory:
      enable_stateful_codec_decoder: false
```

`incremental_codec_cuda_graph`, `incremental_codec_compile` and
`followup_batch_wait_ms` are the individual switches. Measured on one H100
80GB at 20 requests per second, three client seeds of roughly 1200 requests
each: the default path holds 0.6% to 2.3% of streams underrun against 20.9%
for the left-context decoder, with first playable audio at 55 to 58 ms
against 82 to 89 ms.

#### First-audio chunk ramp

For latency-sensitive deployments the whole early chunk schedule can be
configured server-side with `stream_chunk_ramp` on the vocoder stage: entry
`i` sizes streaming decode chunk `i + 1` in codec frames, and past the ramp
the steady stride takes over, so `[2, 4, 8]` yields a
`2 -> 4 -> 8 -> 8 -> ...` schedule. Set it through a pipeline config file:

```yaml
config_cls: VoicingTTSPipelineConfig
model_path: checkpoints/voicing-tts-12hz-0.6b-base
stages:
  vocoder:
    factory:
      stream_chunk_ramp: [2, 4, 8]
```

```bash
python -m sglang_omni.cli serve --config voicing_tts_ramp.yaml
```

Smaller early chunks lower time-to-first-audio but start playback with less
buffered audio, so the continuity cost grows with concurrency. The default
`[1, 2, 4]` is the most aggressive schedule and suits low concurrency; prefer
`[2, 4, 8]` at moderate concurrency and `[4, 8]` for saturated serving, where
the wider first chunk buys back the playback cushion. The ramp is mutually
exclusive with the legacy `initial_chunk_frames` /
`stream_initial_followup_stride` options, its first entry must not exceed the
steady stride, and a per-request `initial_codec_chunk_frames` still overrides
only the first chunk.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | served model | Served model identifier |
| `input` | (required) | Text to synthesize |
| `voice` | `default` | Voice identifier. For Base reference cloning, the reference clip provides the speaker conditioning |
| `references` | `null` | Reference clip for cloning. Each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `language` | `auto` | Target-language hint (see list above) |
| `temperature` | `0.9` | Sampling temperature |
| `top_p` | `1.0` | Top-p sampling |
| `top_k` | `50` | Top-k sampling |
| `repetition_penalty` | `1.05` | Repetition penalty |
| `max_new_tokens` | `2048` | Maximum number of generated codec tokens |
| `seed` | `null` | Random seed for reproducibility |
| `stream` | `false` | Stream raw PCM audio chunks |
| `initial_codec_chunk_frames` | ramp `1 -> 2 -> 4` when omitted | First streaming vocoder chunk size in codec frames. An explicit value replaces the ramp's first chunk only. Smaller values lower TTFA but underrun more easily; `0` uses the steady stride from the start |
| `stream_codec_output` | `true` | Forward codec frames to the vocoder as they are generated. Set `false` to restore whole-utterance decoding for CustomVoice/VoiceDesign |
| `suppress_bootstrap_silence` | `true` | Withhold the silent bootstrap codec frame's audio from streamed CustomVoice output (validated voice/language pairs only, guarded by a runtime silence check). Set `false` to keep the leading silence |

## Model Variants

| Converted checkpoint | Variant | Parameters | Config |
|---|---|---|---|
| `checkpoints/voicing-tts-12hz-0.6b-base` | Base | 0.6B | `examples/configs/voicing_tts_0_6b.yaml` |
| `checkpoints/voicing-tts-12hz-1.7b-base` | Base | 1.7B | `examples/configs/voicing_tts_1_7b.yaml` |
| `checkpoints/voicing-tts-12hz-0.6b-customvoice` | CustomVoice | 0.6B | `examples/configs/voicing_tts_0_6b_customvoice.yaml` |
| `checkpoints/voicing-tts-12hz-1.7b-customvoice` | CustomVoice | 1.7B | `examples/configs/voicing_tts_1_7b_customvoice.yaml` |
| `checkpoints/voicing-tts-12hz-1.7b-voicedesign` | VoiceDesign | 1.7B | `examples/configs/voicing_tts_1_7b_voicedesign.yaml` |

Ascend NPU variants of these configs carry an `_npu` suffix. Both Base sizes expose an identical
request API. The 1.7B model has higher capacity (typically better
quality) at a larger memory and latency cost; the 0.6B model is lighter and faster.

## CustomVoice Checkpoints

CustomVoice generates speech with built-in speakers through the same pipeline. Use it without reference audio; omit `ref_audio`, `ref_text`, `references`, and `x_vector_only_mode`. Omit `task_type` or set it to `CustomVoice`.

| Converted checkpoint | Config | Instruction guidance |
|---|---|---|
| `checkpoints/voicing-tts-12hz-0.6b-customvoice` | `examples/configs/voicing_tts_0_6b_customvoice.yaml` | Accepted for backward compatibility, but not recommended |
| `checkpoints/voicing-tts-12hz-1.7b-customvoice` | `examples/configs/voicing_tts_1_7b_customvoice.yaml` | Supported |

Both converted checkpoints provide `Serena`, `Vivian`, `Uncle_Fu`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`, `Eric`, and `Dylan`. Speaker matching is case-insensitive; an omitted or `default` voice selects `Vivian`. `GET /v1/audio/voices` lists `default` and the served checkpoint's speakers. Unknown speakers or supplied cloning fields return HTTP 400; uploaded reference voices are not used for CustomVoice synthesis.

Both sizes support buffered speech, batch requests, incremental HTTP PCM output, and WebSocket audio output. HTTP streaming requires `stream=true` with `response_format="pcm"`; WebSocket sessions use `stream_audio=true` with `response_format="pcm"`.

**0.6B instruction compatibility:** SGLang-Omni continues to pass optional `instructions` into the 0.6B prompt, preserving existing behavior. The released 0.6B model does not provide reliable instruction control; omit this field or use 1.7B when style control is needed.

**Eric/Dylan language behavior:** For both sizes, `language: Auto` selects Eric's Sichuan dialect token or Dylan's Beijing dialect token. An explicit language takes precedence: `language: Chinese` keeps the Chinese language token. This preserves existing SGLang-Omni behavior. This is a conditioning choice, not a guarantee that the speaker's accent disappears.

Start the 1.7B checkpoint with its matching config:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b_customvoice.yaml \
  --port 8000
```

Then select a built-in speaker in the request. For 0.6B, use its config from the table and omit `instructions`.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "input": "SGLang-Omni serves Voicing-TTS CustomVoice.",
    "voice": "Ryan",
    "language": "English",
    "instructions": "Speak clearly and calmly."
  }' \
  --output custom-voice.wav
```

## VoiceDesign Checkpoint

VoiceDesign (1.7B only) builds the voice from a text description instead of a reference clip or
a built-in speaker. Requests set `task_type` to `VoiceDesign` and pass a non-empty
`instructions` string:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b_voicedesign.yaml \
  --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "voice": "default",
    "input": "Hello, how are you?",
    "task_type": "VoiceDesign",
    "instructions": "A warm, natural young adult voice."
  }' \
  --output voice-design.wav
```

## Benchmark Results

The results below were measured on the same weights and pipeline before the package was renamed
to Voicing-TTS. Scoring WER now requires an external OpenAI-compatible ASR server, because this
repository no longer hosts ASR models; `benchmarks/tasks/asr.py` is the client.

### 0.6B Base

Voicing-TTS 0.6B Base on Seed-TTS EN (1088 utterances, reference voice cloning from each
prompt), concurrency 16, WER scored with Whisper-large-v3. Hardware: 1× H200 SXM.

| Metric | Value |
|---|---|
| WER (corpus, excl. runaway outliers) | 1.07% |
| WER (per-sample median / p95) | 0.00% / 9.09% |
| WER (corpus micro-avg, raw) | 18.29% |
| Runaway samples (>50% WER) | 2 / 1088 (0.2%) |
| Latency mean / median (s) | 6.61 / 6.24 |
| RTF mean / median | 1.51 / 1.48 |
| Output throughput (tok/s) | 115.4 |
| Completed / failed requests | 1088 / 0 |

Typical output is clean (0.00% median WER, 9.09% p95). Two utterances (0.2%) ran away into a
repetition loop and generated ~164 s of looping audio up to `max_new_tokens`, which alone lifts
the raw micro-average to 18.29%; excluding those, corpus WER is 1.07%. RTF > 1 reflects the
0.6B codec pipeline at concurrency 16, not single-stream latency. The 1.7B checkpoint trades
latency for quality.

### 1.7B CustomVoice

Voicing-TTS 1.7B CustomVoice on the full Seed-TTS-Eval EN and ZH splits, concurrency 16, with 16 warmup requests per language/mode and `max_new_tokens=2048`. EN used Ryan/English and ZH used Vivian/Chinese, without reference audio or instructions. WER/CER was scored with Qwen3-ASR-1.7B at concurrency 32. Hardware: 1× H200 141 GB, BF16, TP1. Sampling overrides and seed were unset.

The server used `--tts_engine.engine.max_running_requests 64`, `--tts_engine.engine.cuda_graph_max_bs 64`, `--tts_engine.engine.torch_compile_max_bs 64`, `--vocoder.process vocoder`, `--tts_engine.gpu_memory_fraction 0.85`, and `--vocoder.gpu_memory_fraction 0.10`; `torch.compile` remained disabled. Streaming used the default `1 -> 2 -> 4` chunk ramp without a request-level override. Each language/mode was measured once, in non-streaming EN/ZH then streaming EN/ZH order on the same warmed server. The target GPU had no external GPU process during timed windows; host CPU, memory, and I/O were shared with another profiling task.

| Metric | Non-streaming EN | Non-streaming ZH | Streaming EN | Streaming ZH |
|---|---:|---:|---:|---:|
| Samples | 1088 | 2020 | 1088 | 2020 |
| Corpus WER/CER | 1.608% | 0.984% | 2.085% | 0.927% |
| Corpus WER/CER (excl. >50% outliers) | 1.359% | 0.984% | 1.454% | 0.927% |
| Samples above 50% WER/CER | 3 | 0 | 4 | 0 |
| UTMOS | 4.1723 | 3.1824 | 4.1500 | 3.1789 |
| QPS | 14.788 | 13.307 | 10.098 | 8.333 |
| Latency mean (s) | 1.075 | 1.198 | 1.573 | 1.915 |
| RTF mean | 0.2335 | 0.2079 | 0.3380 | 0.3332 |
| TTFA mean (s) | N/A | N/A | 0.1213 | 0.1047 |

Corpus WER (EN) / CER (ZH) is total edit distance divided by total reference words / characters and includes every sample. The filtered row excludes samples whose own WER/CER exceeds 50% and recomputes the corpus rate. UTMOS is the mean predicted audio-quality score, not a listening-test score. These independently sampled runs are not a paired comparison of streaming and non-streaming quality; the Base result also uses different conditioning and a different ASR evaluator.

TTFA measures arrival of the first PCM payload, not the first audible speech; its mean payload duration was 80 ms in both languages. All 3,108 streaming requests were continuity-scored: 98.99% EN and 93.76% ZH had no playback underrun longer than 50 ms. Maximum underrun was 396.1 ms EN and 2072.0 ms ZH. The default ramp therefore does not guarantee uninterrupted playback at concurrency 16; see [First-audio chunk ramp](#first-audio-chunk-ramp) for the buffering tradeoff.

## Known Limitations

- **Reference audio recommended.** As a cloning model, Voicing-TTS Base produces robotic speech
  without a reference clip.
- **Transcript improves cloning.** Providing `text` in `references` (ICL mode) yields better
  speaker similarity than speaker-embedding-only (x-vector) mode.
- **Language detection.** `language: auto` may misdetect for short or code-switched inputs;
  set `language` explicitly when you know the target language.
- **Rare runaway generation.** Roughly 0.2% of utterances (observed on the 0.6B checkpoint) can
  fall into a repetition loop and keep generating up to `max_new_tokens`. Raising
  `repetition_penalty` (default `1.05`) or lowering `max_new_tokens` mitigates it; the 1.7B
  checkpoint is less prone.
