# TTS Model Usage

This guide covers the `/v1/audio/speech` API with [Voicing-TTS](../cookbook/voicing_tts.md), the model SGLang-Omni serves. The examples use the Base checkpoint, which clones a voice from a reference clip; the CustomVoice and VoiceDesign checkpoints use the same endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md). Voicing-TTS needs no extra packages.

Convert a checkpoint before serving it. The converter accepts a Qwen3-TTS 12Hz checkpoint directory or Hugging Face repo id and writes a Voicing-TTS checkpoint:

```bash
python -m sglang_omni.models.voicing_tts.convert_checkpoint \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  checkpoints/voicing-tts-12hz-1.7b-base
```

See [Convert a Checkpoint](../cookbook/voicing_tts.md#convert-a-checkpoint) for every variant and its target directory. Keep `voicing-tts` in a Base checkpoint's directory name and end it with `base`; uploaded reference voices are enabled from that path.

## Voicing-TTS Variants

| Variant | Example config | Request notes |
|---|---|---|
| [Base](../cookbook/voicing_tts.md#voice-cloning) | `examples/configs/voicing_tts_0_6b.yaml`, `examples/configs/voicing_tts_1_7b.yaml` | Requires reference audio through `ref_audio`, `references[0].audio_path`, or an uploaded voice. `language` defaults to `auto` |
| [CustomVoice](../cookbook/voicing_tts.md#customvoice-checkpoints) | `examples/configs/voicing_tts_0_6b_customvoice.yaml`, `examples/configs/voicing_tts_1_7b_customvoice.yaml` | Text-only synthesis with built-in speakers; omit `voice` for Vivian. Both sizes support streaming; use 1.7B for instruction control |
| [VoiceDesign](../cookbook/voicing_tts.md#voicedesign-checkpoint) | `examples/configs/voicing_tts_1_7b_voicedesign.yaml` | Requires `task_type="VoiceDesign"` and non-empty `instructions`. No reference audio is required |

Ascend NPU variants of these configs carry an `_npu` suffix.

## Launch the Server

See [TTS Process Topology](tts_process_topology.md) for per-stage `process`
overrides and same-GPU memory requirements.

The reference-audio examples below fetch clips from Hugging Face. Remote
references from any public host are allowed by default; to restrict them, pass
`--allowed-media-domain` once per allowed host, as below. Omit those flags when
your requests use only text, uploaded voices, local/file references, or data
URLs.

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

Each example config points `model_path` at its converted checkpoint directory.
Pass `--model-path` to serve a checkpoint stored elsewhere.

Batch speech requests accept up to 32 items by default. Use
`--tts-batch-max-items` to change the server-side request envelope limit:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b.yaml \
  --tts-batch-max-items 32 \
  --port 8000
```

For CustomVoice:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b_customvoice.yaml \
  --port 8000
```

For 0.6B CustomVoice, use `examples/configs/voicing_tts_0_6b_customvoice.yaml`.

For VoiceDesign:

```bash
sgl-omni serve \
  --config examples/configs/voicing_tts_1_7b_voicedesign.yaml \
  --port 8000
```

## Use Curl

The examples below use a sample clip from [`seed-tts-eval-mini`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini). The `references` field accepts `audio_path` (a local path, file URL, data URL, or HTTP URL) and `text` (transcript of that audio). Supplying the transcript enables in-context-learning (ICL) mode, which clones the voice more closely than speaker-embedding (x-vector) mode.

### Voice Cloning (Base)

1. Non-streaming request

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are shorthand for `references[0].audio_path` and
`references[0].text`:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

2. Streaming

Enable streaming to receive raw PCM audio chunks in real time. HTTP streaming
requires both `"stream": true` and `"response_format": "pcm"`:

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

Streaming returns 16-bit mono PCM bytes (`audio/pcm`) with sample-rate metadata
in response headers. It does not include in-band JSON events, final usage, or a
terminal sentinel. When the client does not set `initial_codec_chunk_frames`,
Voicing-TTS ramps its first chunks `1 -> 2 -> 4` codec frames before the steady
stride. Set the field explicitly to override the first chunk, or set it to `0`
to use the steady chunk size from the start.

### CustomVoice

CustomVoice uses a built-in speaker without reference audio:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "input": "Hello from Voicing-TTS CustomVoice.",
    "voice": "Ryan",
    "language": "English",
    "instructions": "Speak clearly and calmly."
  }' \
  --output custom-voice.wav
```

Omit cloning fields (`ref_audio`, `ref_text`, `references`, and `x_vector_only_mode`) and omit `task_type` or set it to `CustomVoice`. For 0.6B, omit `instructions`: it remains accepted for compatibility, but reliable instruction control is not supported. See [CustomVoice checkpoints](../cookbook/voicing_tts.md#customvoice-checkpoints) for speaker discovery, streaming, and Eric/Dylan language behavior.

### VoiceDesign

VoiceDesign uses text plus voice instructions:

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
    --output output.wav
```

### Batch Speech

Use `/v1/audio/speech/batch` when one request should synthesize several
independent utterances. Batch defaults are merged with each item. Item fields
override the defaults, and each item runs through the normal `/v1/audio/speech`
path. With a Base checkpoint, a batch-level reference clip applies to every item
that does not bring its own:

```bash
curl -X POST http://localhost:8000/v1/audio/speech/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "voice": "default",
    "response_format": "wav",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "items": [
      {"input": "First sentence."},
      {"input": "Second sentence.", "speed": 1.1},
      {"input": "Third sentence.", "language": "English"}
    ]
  }'
```

The response preserves item order. Successful items contain base64-encoded
audio bytes and the selected media type. Failed items contain an OpenAI-style
error object at the item level. Invalid batch envelopes, such as too many
items, fail the HTTP request.

### WebSocket Speech Streaming

Use `/v1/audio/speech/stream` for stateful text input over a persistent
WebSocket. The first message must be `session.config`. Then send `input.text`
messages. Send `input.commit` to flush the current text segment while keeping
the WebSocket open, or finish the session with `input.done`. The server
acknowledges the initial configuration with `session.configured`.

`stream_audio` defaults to `false`. With the default, each completed text
segment returns one binary audio frame between `audio.start` and `audio.done`.
For `stream_audio=true`, `response_format` must be `pcm`, and the server sends
incremental binary PCM frames between `audio.start` and `audio.done`.

```python
import asyncio
import json

import websockets

REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."


async def main():
    async with websockets.connect(
        "ws://localhost:8000/v1/audio/speech/stream"
    ) as ws:
        await ws.send(json.dumps({
            "type": "session.config",
            "session": {
                "model": "voicing-tts",
                "voice": "default",
                "ref_audio": REFERENCE_AUDIO,
                "ref_text": REFERENCE_TEXT,
                "response_format": "pcm",
                "stream_audio": True,
                "split_granularity": "sentence",
            },
        }))
        print(await ws.recv())

        pcm_chunks = []
        await ws.send(json.dumps({
            "type": "input.text",
            "text": "Hello from the speech WebSocket. This is the second sentence.",
        }))
        await ws.send(json.dumps({"type": "input.commit"}))

        # input.committed is emitted after all audio for the segment. More
        # input.text/input.commit pairs can follow on the same WebSocket.
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm_chunks.append(message)
                continue
            event = json.loads(message)
            print(event)
            if event["type"] == "input.committed":
                break

        await ws.send(json.dumps({"type": "input.done"}))

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm_chunks.append(message)
                continue
            event = json.loads(message)
            print(event)
            if event["type"] == "session.done":
                break

        with open("websocket_output.pcm", "wb") as f:
            f.write(b"".join(pcm_chunks))


asyncio.run(main())
```

Each `input.commit` forces a flush of any remaining buffered text (including text that does not end at the configured sentence
or clause boundary). After all audio for that segment, the server emits
`input.committed` with `segment_index`, `segment_sentences`, and cumulative
`total_sentences`, then accepts more input on the same connection. An empty
commit is valid: it flushes nothing and reports `segment_sentences: 0`.
`input.done` performs the same final buffer flush, emits `session.done`, and
closes the WebSocket.

`split_granularity` can be `sentence` or `clause`. Unknown message types and
malformed JSON return a WebSocket `error` event. Missing or invalid initial
configuration returns an error and closes the session.

### Uploaded Voices

A Base checkpoint can register reference clips once through `/v1/audio/voices`
and reuse them by name in later `/v1/audio/speech` requests. Uploaded samples
are stored as `.safetensors` files under `SPEAKER_SAMPLES_DIR` and are restored
when the server restarts. If `SPEAKER_SAMPLES_DIR` is not set, the server uses
`~/.cache/sglang-omni/speakers`. `SPEAKER_MAX_UPLOADED` limits the number of
stored voices and defaults to `1000`.

The server enables uploaded voices only when the served checkpoint path
identifies a Base checkpoint: a path component containing `voicing-tts` that
ends in `base`, such as `checkpoints/voicing-tts-12hz-1.7b-base`. On Base, a
request without reference fields whose `voice` is neither `default` nor an
uploaded name returns HTTP 400.

Upload a voice sample:

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -F "name=narrator" \
  -F "consent=consent-recording-id" \
  -F "ref_text=Transcript of the uploaded reference clip." \
  -F "speaker_description=Clear narration voice" \
  -F "audio_sample=@reference.wav;type=audio/wav"
```

List preset and uploaded voices:

```bash
curl http://localhost:8000/v1/audio/voices
```

For CustomVoice, the response's `voices` list contains `default` and the served checkpoint's built-in speakers. Uploaded reference voices are not used for CustomVoice synthesis.

Use the uploaded voice by name:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "input": "The uploaded voice can now be reused without resending audio.",
    "voice": "narrator",
    "response_format": "wav"
  }' \
  --output narrator.wav
```

Delete an uploaded voice:

```bash
curl -X DELETE http://localhost:8000/v1/audio/voices/narrator
```

Accepted upload formats are WAV, MP3, FLAC, OGG, AAC, WebM, and MP4. Each file
must be at most 10 MiB and contain 1-30 seconds of non-silent reference audio.
Uploading the same `name` overwrites the previous sample. Deleting a voice
removes the persisted sample. The list response includes API-process
`cache_stats` for uploaded-voice reference lookup observability.

## Use Python

```python
REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."
```

### Voice Cloning

1. Non-streaming Request

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "voicing-tts",
        "voice": "default",
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

2. Streaming Request

```python
import wave

import requests

payload = {
    "model": "voicing-tts",
    "voice": "default",
    "input": SPEECH_INPUT,
    "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    "stream": True,
    "response_format": "pcm",
}

chunks = []
with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    sample_rate = int(stream.headers.get("x-sample-rate", 24000))
    for chunk in stream.iter_content(chunk_size=None):
        if chunk:
            chunks.append(chunk)

with wave.open("output_stream.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate or 24000)
    w.writeframes(b"".join(chunks))
```

### OpenAI Python SDK

The endpoint is compatible with the OpenAI Python SDK when the client points to
the SGLang-Omni server. Pass the reference clip through `extra_body`, or use an
uploaded voice name as `voice`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.audio.speech.create(
    model="voicing-tts",
    voice="default",
    input=SPEECH_INPUT,
    response_format="wav",
    extra_body={"ref_audio": REFERENCE_AUDIO, "ref_text": REFERENCE_TEXT},
)
response.stream_to_file("output.wav")
```

## Request Parameters

The table below lists all parameters accepted by the `/v1/audio/speech` endpoint.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served model identifier |
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Built-in CustomVoice speaker or uploaded Base voice |
| `response_format` | string | `"wav"` | Output audio format: `wav`, `mp3`, `flac`, `pcm`, `aac`, or `opus` |
| `speed` | float | `1.0` | Playback speed multiplier from `0.25` to `4.0` |
| `stream` | bool | `false` | Enable raw PCM streaming. When true, `response_format` must be `pcm` |
| `initial_codec_chunk_frames` | int | `null` | Optional first codec chunk size for streaming TTFA / playback-continuity tuning. When omitted, Voicing-TTS ramps `1 -> 2 -> 4` into the steady stride. An explicit `0` uses the steady chunk size from the start |
| `stream_codec_output` | bool | `true` | Forward codec frames to the vocoder as they are generated. Set `false` to restore whole-utterance decoding for CustomVoice / VoiceDesign |
| `suppress_bootstrap_silence` | bool | `true` | Withhold the silent bootstrap codec frame's audio from streamed CustomVoice output on validated voice/language pairs; an audible first frame is always emitted unchanged. Set `false` to keep the leading silence |
| `references` | list | `null` | Reference audio for voice cloning. Each item has `audio_path` (local path / file URL / data URL / remote URL) and `text` |
| `ref_audio` | string | `null` | Reference audio path / URL / base64 string. Equivalent to `references[0].audio_path` |
| `ref_text` | string | `null` | Transcript for `ref_audio`. Equivalent to `references[0].text` |
| `language` | string | `null` | Language hint: `Auto`, `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, or `Italian` |
| `task_type` | string | `null` | `Base`, `CustomVoice`, or `VoiceDesign`. Inferred as `Base` when reference audio/text is present, otherwise `CustomVoice` |
| `instructions` | string | `null` | CustomVoice style or VoiceDesign instructions |
| `max_new_tokens` | int | `null` | Maximum number of generated codec tokens |
| `x_vector_only_mode` | bool | `null` | Base speaker-embedding mode |
| `temperature` | float | `null` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `repetition_penalty` | float | `null` | Repetition penalty |
| `seed` | int | `null` | Request-scoped random seed |

Invalid speech requests return an OpenAI-style error envelope:

```json
{
  "error": {
    "message": "stream=true requires response_format='pcm'",
    "type": "BadRequestError",
    "param": "response_format",
    "code": 400
  }
}
```

## H200 SeedTTS Benchmark Commands

Download the full SeedTTS set first:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
```

Run EN and ZH after launching the target server on port 8000. WER scoring needs
an external OpenAI-compatible ASR server, because this repository no longer
hosts ASR models; `benchmarks/tasks/asr.py` is the client. Do not add benchmark
results to docs until the full H200 runs complete.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model checkpoints/voicing-tts-12hz-0.6b-base \
  --port 8000 \
  --output-dir results/voicing_tts_0_6b_en \
  --lang en \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model checkpoints/voicing-tts-12hz-0.6b-base \
  --port 8000 \
  --output-dir results/voicing_tts_0_6b_zh \
  --lang zh \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model checkpoints/voicing-tts-12hz-1.7b-base \
  --port 8000 \
  --output-dir results/voicing_tts_1_7b_en \
  --lang en \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model checkpoints/voicing-tts-12hz-1.7b-base \
  --port 8000 \
  --output-dir results/voicing_tts_1_7b_zh \
  --lang zh \
  --max-concurrency 16
```
