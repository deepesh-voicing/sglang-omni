# API Server Design

This page explains the API server at the level that is most useful for maintenance: where it sits in the system, which files matter, and how requests are mapped into the runtime.

If you only want to launch the server and call it, start with the [Voicing-TTS cookbook](../cookbook/voicing_tts.md).

## Role in the System

The API server is the outer protocol layer on top of the `sglang-omni` pipeline runtime.

At a high level, the built-in server startup path is:

`CLI / Python entrypoint` → `PipelineConfig` → `Pipeline Startup` → `Coordinator` → `Client` → `FastAPI`

After startup, the request path is:

`HTTP request` → `FastAPI route` → `Client` → `Coordinator` → `Stage pipeline` → `Client aggregation` → `HTTP response`

That split keeps responsibilities clean:

- the pipeline runtime handles orchestration and execution
- the `Client` layer submits requests and assembles results
- the API server translates between HTTP/OpenAI-style payloads and those internal abstractions

## Key Files

For the current server implementation, these are the files that matter most.

| File | Role |
| --- | --- |
| `sglang_omni/serve/openai_api.py` | Defines the FastAPI app, routes, request conversion, and response formatting |
| `sglang_omni/serve/protocol.py` | Defines request and response schemas |
| `sglang_omni/serve/speech_service.py` | Validates speech requests and converts them into `GenerateRequest` objects |
| `sglang_omni/serve/speech_ws.py` | Runs `/v1/audio/speech/stream` WebSocket sessions |
| `sglang_omni/serve/launcher.py` | Compiles the pipeline, starts the runtime, mounts the app, and runs Uvicorn |
| `sglang_omni/client/client.py` | Submits requests to the coordinator and aggregates audio and stream results |
| `sglang_omni/cli/serve.py` | Defines the current CLI surface for `sgl-omni serve` |

If you are tracing endpoint behavior, `openai_api.py`, `speech_service.py`, and `client.py` are usually the best places to start.

## `create_app()` vs `launch_server()`

This is the most important distinction in the serving code.

### `create_app(client, model_name=...)`

`create_app()` only builds the FastAPI app and registers the core routes.

It does **not**:

- compile the pipeline
- start the runtime
- create the coordinator
- mount profiling routes
- run Uvicorn

Use it when you already have a live `Client` and want to embed the HTTP layer yourself.

### `launch_server(pipeline_config, ...)`

`launch_server()` is the full built-in server lifecycle.

It:

- compiles the pipeline config
- starts the pipeline runtime
- creates the `Client`
- creates the FastAPI app
- mounts profiling routes on the single-process path
- runs Uvicorn
- stops the runtime on shutdown

Use it when you want the standard out-of-the-box server path.

## Route Surface

The current server exposes these main routes:

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Health status from `client.health()` |
| `GET` | `/v1/models` | Single-model listing for the active pipeline |
| `POST` | `/v1/audio/speech` | Text-to-speech, raw audio response or raw PCM chunks when `stream=true` |
| `POST` | `/v1/audio/speech/batch` | Several speech items in one request, returned as one JSON response |
| `WS` | `/v1/audio/speech/stream` | Streaming speech session over a WebSocket |
| `GET` / `POST` | `/v1/audio/voices` | List or upload reference voices |
| `DELETE` | `/v1/audio/voices/{name}` | Delete an uploaded reference voice |
| `POST` | `/start_profile` | Torch trace + (optional) request-level events. Added by the built-in launcher |
| `POST` | `/stop_profile` | Stops both torch trace and request-level events |
| `POST` | `/start_request_profile` | Request-level event recorder only (no torch trace) |
| `POST` | `/stop_request_profile` | Stops the request-level event recorder |

The profiling routes are mounted by the single-process `launch_server()` path. The
current multi-process launcher path does not mount them.

`/start_profile` accepts:

```jsonc
{
  "run_id": "demo-run",
  "trace_path_template": "/tmp/profiles/demo-run/trace",  // torch trace template
  "event_dir": "/tmp/profiles/demo-run/events",            // request-event JSONL dir (optional)
  "enable_torch": true                                     // set false to skip torch trace
}
```

`/stop_profile` and `/stop_request_profile` both accept an optional
`run_id`. Omitting it is a wildcard: every stage stops whatever profiler
session is currently active.

Request-level events are emitted as JSON lines under
`<event_dir>/events_<stage>_<pid>.jsonl`. Use `python -m sglang_omni.profiler
<event_dir>` to derive the timeline / stage / hop reports described in
`docs/developer_reference/profiler.md`.

## Request Mapping

The server does not pass OpenAI-style request bodies straight into the runtime. It first converts them into internal request objects.

### Conversion into `GenerateRequest`

`SpeechRequestValidator.build_generate_request()` in `speech_service.py` is the key translation point for `CreateSpeechRequest`. It:

- builds `SamplingParams` from the request, with speech defaults for fields the client omits
- builds the prompt from `input` and any reference audio descriptors
- passes per-stage runtime params through `stage_params`
- stores TTS-specific parameters such as `voice`, `task_type`, `language`, `instructions`, `ref_audio`, and `ref_text` under `tts_params` in request metadata
- resolves an uploaded voice name into its stored reference audio
- sets `output_modalities=["audio"]` and `task="tts"` in metadata

The route hands that `GenerateRequest` to `Client`. The client then converts it
to an `OmniRequest` before submitting it to the coordinator.

## Response Paths

### Speech / TTS

For non-streaming requests, `Client.speech()` collects audio chunks, encodes
them, and returns raw audio bytes to the HTTP layer.

For `stream=true`, the route emits `audio/pcm` bytes directly. The HTTP response
headers are derived from the first audio chunk and subsequent chunks must keep
the same sample rate. TTS chunk-timing knobs such as
`initial_codec_chunk_frames` are forwarded as request params only when the
client provides them, including an explicit `0`, so model schedulers can consume
them without changing Stage, Coordinator, or Relay. When the field is omitted,
the speech route leaves it unset and the model applies its own streaming
default.
