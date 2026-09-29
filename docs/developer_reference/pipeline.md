## Pipeline Overview

### Coordinator

`Coordinator` is the global request router. It registers stage endpoints, sends
new requests to the entry stage, receives `CompleteMessage` and `StreamMessage`
events, and resolves client futures or streams.

Key responsibilities:

- route new requests to `entry_stage`
- track request state: pending, running, completed, failed, aborted
- collect terminal stage completions
- merge results when a pipeline has multiple terminal stages
- broadcast abort messages to all stages

The coordinator is stage-implementation agnostic. In a tensor
parallel stage group, it only talks to rank 0. Peer ranks stay internal to the
stage group.

### Stage

`Stage` is an IO shell. It handles all inter-stage communication. It receives control messages, reads
and writes relay payloads, performs fan-in when needed, and pushes all executable
work into `scheduler.inbox`.

```python
class Stage:
    def __init__(
        self,
        name,
        control_plane,
        relay,
        get_next,
        input_handler,
        scheduler,
        stream_targets,
        same_gpu_targets,
    ):
        self.scheduler = scheduler
```

Stage responsibilities:

- receive `SubmitMessage`, `DataReadyMessage`, `ShutdownMessage`, and profiler
  control messages over ZMQ
- receive `AbortMessage` over the coordinator broadcast channel
- read and write full `StagePayload` objects through relay
- aggregate inputs with `AggregatedInput` for fan-in stages
- route normal results to downstream stages or the coordinator
- route streaming chunks, including same-GPU CUDA IPC and cross-GPU relay
- drain `scheduler.outbox` and convert scheduler output into control-plane
  messages

The important invariant is that `Stage` does not branch on scheduler type.
`SimpleScheduler`, `OmniScheduler`, and streaming schedulers all present the
same surface.

### Scheduler

All schedulers implement the same interface:

```python
class Scheduler:
    inbox: Queue[IncomingMessage]
    outbox: Queue[OutgoingMessage]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def abort(self, request_id: str) -> None: ...
```

Scheduler messages are used to communicate with stage layer:

```python
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: Any

class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error"]
    data: Any
    target: str | None
    metadata: dict[str, Any] | None
```

#### OmniScheduler

`OmniScheduler` is used by autoregressive stages. It composes with SGLang's
upstream scheduler. The goal is to reuse SGLang's
batch selection, KV cache management, prefill/decode scheduling, and tree cache
while keeping SGLang-Omni's transport, request objects, and streaming behavior
outside the upstream scheduler. (Overlap scheduling is explicitly unsupported:
`OmniScheduler._event_loop_overlap` refuses to run because the
`Req.inflight_middle_chunks` decrement would lag one iteration on that loop.)

#### SimpleScheduler

`SimpleScheduler` is for non-AR stages such as preprocessing and encoders. It
has no KV cache and no SGLang batching. The loop is:

```text
inbox.get() -> compute function -> outbox.put(result or error)
```

It supports a batch compute function for stages where local batching is
useful.

#### Streaming vocoder schedulers

Streaming vocoder schedulers, such as the Voicing-TTS `vocoder` stage's
`VoicingTTSStreamingVocoderScheduler`, turn codec chunks from an AR stage into
audio. They build on `StreamingVocoderBase` in
`sglang_omni/scheduling/streaming_vocoder.py` and handle:

- `new_request`: initialize per-request state
- `stream_chunk`: accumulate and decode code chunks
- `stream_done`: flush remaining audio and emit a final result

### Model Runner

The model runner layer owns the AR forward path. The design target is:

```text
ForwardBatch -> before/custom forward hooks -> model forward -> post hook -> output processing
```

The shared base runner (`ModelRunner` in `sglang_omni/model_runner/base.py`)
owns common mechanics: `ForwardBatch` construction, sampling, logit processing,
repetition penalty handling, output processing, and conversion into scheduler
output. Subclasses override the phase hooks: `before_prefill`,
`before_decode`, `post_prefill`, and `post_decode`.

#### Feedback AR runners

Some AR models need feedback produced by the previous step inside the same
model runner before they can run the next decode step.
`VoicingTTSModelRunner` in `sglang_omni/models/voicing_tts/model_runner.py`
has this shape: each AR step produces a codec frame, and that frame is
written back into the model's decode buffers before the next step.

The pattern covers self-contained feedback loops only:

- write previous-step feedback into model buffers before forward
- run the AR backbone and secondary head inside model `forward()`
- extract codebook outputs and feedback tensors after forward
- push stream or result output to the scheduler outbox

Cross-stage feedback, where the producer and consumer live in different
schedulers and communicate through relay, is out of scope for this runner.
