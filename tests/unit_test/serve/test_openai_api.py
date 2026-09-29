# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client, ClientError, GenerateChunk
from sglang_omni.client.audio import encode_pcm
from sglang_omni.client.types import GenerateRequest
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import await_speech_response, speech_audio_response
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_service import SpeechRequestValidator
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

MODEL_FAMILIES = {
    "voicing-tts": "vocoder",
}


class FaultInjectingCoordinator(Coordinator):
    """Inject a model-stage failure through the real Coordinator/Client path."""

    def __init__(self, terminal_stage: str, error: str = "cuda out of memory"):
        super().__init__(
            completion_endpoint="inproc://complete",
            abort_endpoint="inproc://abort",
            entry_stage="preprocess",
            terminal_stages=[terminal_stage],
        )
        self.control_plane = RecordingCoordinatorControlPlane()
        self.terminal_stage = terminal_stage
        self.error = error
        self.register_stage("preprocess", "inproc://preprocess")

    async def submit_request(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] | None = None,
    ) -> None:
        await super().submit_request(
            request_id,
            request,
            stream_queue=stream_queue,
        )
        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)
        if bool(request.params.get("stream", False)):
            await self.handle_stream(self.partial_stream_message(request_id, request))
        await self.handle_completion(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.terminal_stage,
                success=False,
                error=self.error,
            )
        )

    def partial_stream_message(
        self, request_id: str, request: OmniRequest
    ) -> StreamMessage:
        if "tts_params" in request.metadata:
            chunk = {
                "audio_data": [0.0, 0.1],
                "sample_rate": 24000,
                "modality": "audio",
            }
            modality = "audio"
        else:
            chunk = {"text": "partial", "modality": "text"}
            modality = "text"
        return StreamMessage(
            request_id=request_id,
            from_stage=self.terminal_stage,
            chunk=chunk,
            stage_name=self.terminal_stage,
            modality=modality,
        )


def fault_client(model_name: str, error: str = "cuda out of memory") -> Client:
    return Client(FaultInjectingCoordinator(MODEL_FAMILIES[model_name], error=error))


class SuccessfulSpeechClient:
    def __init__(
        self, *, sample_rate: int = 24000, finish_reason: str = "stop"
    ) -> None:
        self.sample_rate = sample_rate
        self.finish_reason = finish_reason
        self.generate_requests: list[GenerateRequest] = []
        self.speech_requests: list[GenerateRequest] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        self.generate_requests.append(request)
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=self.sample_rate,
            finish_reason="stop",
        )

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ):
        from sglang_omni.client.types import SpeechResult

        del request_id, speed, allow_format_fallback
        self.speech_requests.append(request)
        return SpeechResult(
            audio_bytes=b"RIFF",
            mime_type=f"audio/{response_format}",
            format=response_format,
            finish_reason=self.finish_reason,
        )


class EmptyStreamingSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=None,
            sample_rate=24000,
            finish_reason="stop",
        )


class FailingSpeechGenerateClient:
    def __init__(self, error: str) -> None:
        self.error = error

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        raise RuntimeError(self.error)
        yield

    async def speech(
        self,
        request: Any,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ):
        del request, request_id, response_format, speed, allow_format_fallback
        raise ClientError(self.error)

    async def abort(self, request_id: str) -> None:
        del request_id


class EmptyDeltaStreamingSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[],
            sample_rate=24000,
            finish_reason=None,
        )
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=None,
            sample_rate=24000,
            finish_reason="stop",
        )


class PrefetchedBlockingStreamingSpeechClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
            finish_reason=None,
        )
        await asyncio.Future()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class BlockingFirstAudioStreamingSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        self.started.set()
        await asyncio.Future()
        yield GenerateChunk(request_id="speech-1")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class BlockingNonStreamingSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ):
        del request, request_id, response_format, speed, allow_format_fallback
        self.started.set()
        await asyncio.Future()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class DisconnectingRequest:
    def __init__(self) -> None:
        self.disconnected = asyncio.Event()

    async def is_disconnected(self) -> bool:
        return self.disconnected.is_set()


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


class BlockingAbortControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.abort_started = asyncio.Event()
        self.release_abort = asyncio.Event()
        self.abort_cancelled = False

    async def broadcast_abort(self, msg: Any) -> None:
        self.aborts.append(msg)
        self.abort_started.set()
        try:
            await self.release_abort.wait()
        except asyncio.CancelledError:
            self.abort_cancelled = True
            raise


def streaming_client(
    control_plane: RecordingCoordinatorControlPlane | None = None,
) -> tuple[Client, Coordinator, RecordingCoordinatorControlPlane]:
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocess",
        terminal_stages=["decode"],
    )
    control_plane = control_plane or RecordingCoordinatorControlPlane()
    coordinator.control_plane = control_plane
    coordinator.register_stage("preprocess", "inproc://preprocess")
    return Client(coordinator), coordinator, control_plane


def http_scope(*, path: str, spec_version: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }


class AdminClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], list[str] | None, float]] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def model_info(
        self,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        self.calls.append(("model_info", {}, stages, timeout_s))
        return {
            "success": True,
            "message": "ok",
            "results": [
                {
                    "stage": "decode",
                    "success": True,
                    "message": "ok",
                    "data": {
                        "model_path": "/tmp/current-model",
                        "load_format": "safetensors",
                        "weight_version": "v1",
                    },
                }
            ],
        }

    async def pause_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        self.calls.append(("pause_generation", payload or {}, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def continue_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        self.calls.append(("continue_generation", payload or {}, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def update_weights_from_disk(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        self.calls.append(("update_weights_from_disk", payload, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def init_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        self.calls.append(("init_weights_update_group", payload, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def destroy_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        self.calls.append(("destroy_weights_update_group", payload, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def update_weights_from_distributed(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        self.calls.append(
            ("update_weights_from_distributed", payload, stages, timeout_s)
        )
        return {"success": True, "message": "ok", "results": []}

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        self.calls.append((action, payload or {}, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def weights_checker(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        self.calls.append(("weights_checker", payload or {}, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}


@pytest.mark.parametrize("model_name", MODEL_FAMILIES)
def test_non_streaming_http_faults_return_500(model_name: str) -> None:
    client = TestClient(create_app(fault_client(model_name), model_name=model_name))

    speech_resp = client.post(
        "/v1/audio/speech",
        json={
            "model": model_name,
            "input": "hello",
            "voice": "default",
            "stream": False,
            "response_format": "wav",
        },
    )
    assert speech_resp.status_code == 500
    assert speech_resp.json()["error"]["type"] == "server_error"
    assert "cuda out of memory" in speech_resp.json()["error"]["message"]


def test_speech_stream_admission_reject_returns_503_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = TestClient(
        create_app(
            FailingSpeechGenerateClient(QueueFullError.MESSAGE),
            model_name="voicing-tts",
        )
    )

    with caplog.at_level(logging.WARNING, logger="sglang_omni.serve.openai_api"):
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "voicing-tts",
                "input": "hello",
                "voice": "default",
                "stream": True,
                "response_format": "pcm",
            },
        )

    assert response.status_code == 503
    assert QueueFullError.MESSAGE in response.json()["error"]["message"]
    assert any(
        rec.levelno == logging.WARNING and "Rejecting speech request" in rec.message
        for rec in caplog.records
    )
    assert not any(rec.exc_info for rec in caplog.records)


@pytest.mark.parametrize("stream", [False, True])
def test_speech_context_rejection_returns_400_without_traceback(
    stream: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    message = "Requested token count exceeds the model's maximum context length"
    client = TestClient(
        create_app(FailingSpeechGenerateClient(message), model_name="voicing-tts")
    )

    with caplog.at_level(logging.WARNING, logger="sglang_omni.serve.openai_api"):
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "voicing-tts",
                "input": "hello",
                "voice": "default",
                "stream": stream,
                "response_format": "pcm" if stream else "wav",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": message,
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    }
    assert any(
        rec.levelno == logging.WARNING and "Rejecting speech request" in rec.message
        for rec in caplog.records
    )
    assert not any(rec.exc_info for rec in caplog.records)


def test_speech_endpoint_rejects_invalid_request_with_openai_error() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts",
            "input": "hello",
            "voice": "default",
            "stream": True,
            "response_format": "wav",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "stream=true requires response_format='pcm'",
            "type": "BadRequestError",
            "param": "response_format",
            "code": 400,
        }
    }


def test_speech_endpoint_returns_binary_audio() -> None:
    speech_client = SuccessfulSpeechClient(finish_reason="length")
    client = TestClient(create_app(speech_client, model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.content == b"RIFF"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-finish-reason"] == "length"
    assert speech_client.speech_requests[0].model == "tts"
    assert speech_client.speech_requests[0].metadata["tts_params"]["voice"] == "default"


def test_create_app_passes_model_specific_speech_input_limit() -> None:
    app = create_app(
        SuccessfulSpeechClient(),
        model_name="voicing-tts",
        max_speech_input_chars=None,
    )

    assert app.state.speech_service.max_speech_input_chars is None


@pytest.mark.parametrize("stream", [False, True])
def test_speech_endpoint_accepts_seedtts_reference_payload_without_voice(
    stream: bool,
) -> None:
    speech_client = SuccessfulSpeechClient()
    client = TestClient(create_app(speech_client, model_name="served-model"))
    ref_audio = base64.b64encode(b"RIFF").decode("ascii")

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "seedtts",
            "input": "hello",
            "ref_audio": f"data:audio/wav;base64,{ref_audio}",
            "ref_text": "reference transcript",
            "response_format": "pcm" if stream else "wav",
            "stream": stream,
        },
    )

    assert response.status_code == 200
    request = (
        speech_client.generate_requests[0]
        if stream
        else speech_client.speech_requests[0]
    )
    assert request.model == "seedtts"
    assert request.metadata["tts_params"]["voice"] == "default"


def test_speech_endpoint_accepts_sdk_shaped_binary_request() -> None:
    speech_client = SuccessfulSpeechClient()
    client = TestClient(create_app(speech_client, model_name="default-tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "voice": "alloy",
            "input": "hello from an SDK-shaped request",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.content == b"RIFF"
    assert response.headers["content-type"] == "audio/wav"
    assert (
        response.headers["content-disposition"] == 'attachment; filename="speech.wav"'
    )
    assert speech_client.speech_requests[0].model == "tts-1"
    assert speech_client.speech_requests[0].metadata["tts_params"]["voice"] == "alloy"


def test_speech_endpoint_rejects_invalid_json_with_openai_error() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        content=b"{",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "BadRequestError"
    assert response.json()["error"]["code"] == 400


def test_speech_endpoint_stream_without_audio_returns_error() -> None:
    client = TestClient(create_app(EmptyStreamingSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts",
            "input": "hello",
            "voice": "default",
            "stream": True,
            "response_format": "pcm",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "No audio output generated" in response.json()["error"]["message"]


def test_speech_endpoint_stream_empty_delta_is_not_success() -> None:
    client = TestClient(create_app(EmptyDeltaStreamingSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts",
            "input": "hello",
            "voice": "default",
            "stream": True,
            "response_format": "pcm",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "No audio output generated" in response.json()["error"]["message"]


def test_admin_routes_forward_to_client() -> None:
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="voicing-tts"))

    info = client.get("/model_info")
    pause = client.post(
        "/pause_generation",
        json={"mode": "in_place", "stages": ["decode"], "timeout_s": 5},
    )
    update = client.post(
        "/update_weights_from_disk",
        json={
            "model_path": "/tmp/new-model",
            "load_format": "safetensors",
            "weight_version": "v2",
            "abort_all_requests": True,
        },
    )
    checksum = client.post("/weights_checker", json={"action": "checksum"})

    assert info.status_code == 200
    assert info.json()["weight_version"] == "v1"
    assert info.json()["model_path"] == "/tmp/current-model"
    assert info.json()["load_format"] == "safetensors"
    assert info.json()["stages"][0]["stage"] == "decode"
    assert pause.status_code == 200
    assert update.status_code == 200
    assert checksum.status_code == 200
    assert admin.calls == [
        ("model_info", {}, None, 30.0),
        ("pause_generation", {"mode": "in_place"}, ["decode"], 5),
        (
            "update_weights_from_disk",
            {
                "model_path": "/tmp/new-model",
                "load_format": "safetensors",
                "abort_all_requests": True,
                "weight_version": "v2",
                "is_async": False,
                "torch_empty_cache": False,
                "keep_pause": False,
                "recapture_cuda_graph": False,
                "token_step": 0,
                "flush_cache": True,
            },
            None,
            120.0,
        ),
        ("weights_checker", {"action": "checksum"}, None, 120.0),
    ]


def test_speech_stream_defaults_to_raw_pcm() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="voicing-tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "voicing-tts",
            "input": "hello",
            "voice": "default",
            "stream": True,
            "response_format": "pcm",
        },
    )

    expected = encode_pcm([0.0, 0.1, -0.1, 0.0], sample_rate=24000)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-channels"] == "1"
    assert response.headers["x-bit-depth"] == "16"
    assert response.content == expected


def test_speech_stream_headers_use_chunk_sample_rate() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(sample_rate=44100), model_name="voicing-tts")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "voicing-tts",
            "input": "hello",
            "voice": "default",
            "stream": True,
            "response_format": "pcm",
        },
    )

    expected = encode_pcm([0.0, 0.1, -0.1, 0.0], sample_rate=44100)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-sample-rate"] == "44100"
    assert response.headers["x-channels"] == "1"
    assert response.headers["x-bit-depth"] == "16"
    assert response.content == expected


def test_raw_pcm_response_close_aborts_inner_speech_stream() -> None:
    async def drive() -> None:
        client = PrefetchedBlockingStreamingSpeechClient()
        response = await speech_audio_response(
            request=ConnectedRequest(),
            client=client,
            gen_req=GenerateRequest(model="voicing-tts", prompt="hello", stream=True),
            request_id="req-1",
            speed=1.0,
        )
        body = response.body_iterator
        assert await anext(body) == encode_pcm([0.0, 0.1, -0.1, 0.0], 24000)
        await body.aclose()
        assert client.aborted == ["req-1"]

    asyncio.run(drive())


def test_raw_pcm_response_disconnect_before_first_chunk_aborts_request() -> None:
    async def drive() -> None:
        client = BlockingFirstAudioStreamingSpeechClient()
        request = DisconnectingRequest()
        task = asyncio.create_task(
            speech_audio_response(
                request=request,
                client=client,
                gen_req=GenerateRequest(
                    model="voicing-tts", prompt="hello", stream=True
                ),
                request_id="req-1",
                speed=1.0,
            )
        )
        await client.started.wait()
        request.disconnected.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.aborted == ["req-1"]

    asyncio.run(drive())


def test_speech_stream_rejects_non_pcm_response_format() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="voicing-tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "voicing-tts",
            "input": "hello",
            "voice": "default",
            "stream": True,
            "response_format": "wav",
        },
    )

    assert 400 <= response.status_code < 500
    assert "response_format" in response.text
    assert "pcm" in response.text.lower()


def test_speech_request_carries_initial_codec_chunk_frames() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
        initial_codec_chunk_frames=4,
    )

    gen_req = SpeechRequestValidator(
        default_model="voicing-tts"
    ).build_generate_request(req)

    assert gen_req.extra_params["initial_codec_chunk_frames"] == 4


def test_raw_pcm_speech_request_defers_initial_chunk_to_model() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
    )

    gen_req = SpeechRequestValidator(
        default_model="voicing-tts"
    ).build_generate_request(req)

    assert "initial_codec_chunk_frames" not in gen_req.extra_params


def test_raw_pcm_speech_request_respects_explicit_initial_zero() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
        initial_codec_chunk_frames=0,
    )

    gen_req = SpeechRequestValidator(
        default_model="voicing-tts"
    ).build_generate_request(req)

    assert gen_req.extra_params["initial_codec_chunk_frames"] == 0


def test_speech_response_disconnect_aborts_active_request() -> None:
    async def drive() -> None:
        client = BlockingNonStreamingSpeechClient()
        request = DisconnectingRequest()
        task = asyncio.create_task(
            await_speech_response(
                request=request,
                client=client,
                gen_req=GenerateRequest(model="voicing-tts", prompt="hello"),
                request_id="req-1",
                response_format="wav",
                speed=1.0,
            )
        )
        await client.started.wait()
        request.disconnected.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.aborted == ["req-1"]

    asyncio.run(drive())


def test_speech_response_returns_when_disconnect_poll_is_false() -> None:
    async def drive() -> None:
        result = await await_speech_response(
            request=ConnectedRequest(),
            client=SuccessfulSpeechClient(),
            gen_req=GenerateRequest(model="voicing-tts", prompt="hello"),
            request_id="req-1",
            response_format="wav",
            speed=1.0,
        )
        assert result.audio_bytes == b"RIFF"

    asyncio.run(drive())


def test_speech_request_records_explicit_generation_params() -> None:
    req = CreateSpeechRequest(
        input="hello",
        temperature=0.8,
        top_k=30,
        seed=123,
    )

    gen_req = SpeechRequestValidator(default_model="qwen3-tts").build_generate_request(
        req
    )

    assert gen_req.sampling.temperature == 0.8
    assert gen_req.sampling.top_k == 30
    assert gen_req.sampling.seed == 123
    assert gen_req.metadata["tts_params"]["explicit_generation_params"] == [
        "seed",
        "temperature",
        "top_k",
    ]


def test_speech_request_passes_streaming_control_fields() -> None:
    req = CreateSpeechRequest(
        input="hello",
        initial_codec_chunk_frames=8,
        x_vector_only_mode=True,
        response_format="pcm",
        stream=True,
    )

    gen_req = SpeechRequestValidator(default_model="qwen3-tts").build_generate_request(
        req
    )
    tts_params = gen_req.metadata["tts_params"]

    assert tts_params["initial_codec_chunk_frames"] == 8
    assert tts_params["x_vector_only_mode"] is True
    assert tts_params["response_format"] == "pcm"
    assert gen_req.extra_params == {"initial_codec_chunk_frames": 8}


# ---------------------------------------------------------------------------
# Admin auth tests
# ---------------------------------------------------------------------------

ADMIN_PATHS_THAT_NEED_AUTH = [
    ("GET", "/model_info"),
    ("POST", "/model_info"),
    ("POST", "/pause_generation"),
    ("POST", "/continue_generation"),
    ("POST", "/update_weights_from_disk"),
    ("POST", "/update_weights_from_tensor"),
    ("POST", "/update_weights_from_distributed"),
    ("POST", "/init_weights_update_group"),
    ("POST", "/destroy_weights_update_group"),
    ("GET", "/weights_checker"),
    ("POST", "/weights_checker"),
]

ADMIN_API_KEY = "secret-key"


def admin_headers(
    key: str = ADMIN_API_KEY,
    *,
    scheme: str = "Bearer",
) -> dict[str, str]:
    return {"Authorization": f"{scheme} {key}"}


def test_admin_routes_open_when_no_key_configured() -> None:
    """Without a key, all admin routes are accessible with no auth header."""
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="voicing-tts"))

    resp = client.get("/model_info")
    assert resp.status_code == 200

    resp = client.post("/pause_generation", json={})
    assert resp.status_code == 200


def test_admin_routes_require_bearer_token_when_key_configured() -> None:
    """When admin_api_key is set, requests without the header are rejected."""
    admin = AdminClient()
    client = TestClient(
        create_app(admin, model_name="voicing-tts", admin_api_key=ADMIN_API_KEY)
    )

    for method, path in ADMIN_PATHS_THAT_NEED_AUTH:
        resp = client.request(method, path, json={})
        assert (
            resp.status_code == 401
        ), f"{method} {path} should be 401, got {resp.status_code}"
        assert "WWW-Authenticate" in resp.headers


def test_admin_routes_reject_wrong_bearer_token() -> None:
    admin = AdminClient()
    client = TestClient(
        create_app(admin, model_name="voicing-tts", admin_api_key=ADMIN_API_KEY)
    )

    for method, path in ADMIN_PATHS_THAT_NEED_AUTH:
        resp = client.request(method, path, json={}, headers=admin_headers("wrong-key"))
        assert (
            resp.status_code == 403
        ), f"{method} {path} should be 403, got {resp.status_code}"


def test_admin_routes_accept_correct_bearer_token() -> None:
    admin = AdminClient()
    client = TestClient(
        create_app(admin, model_name="voicing-tts", admin_api_key=ADMIN_API_KEY)
    )

    resp = client.get("/model_info", headers=admin_headers(scheme="bearer"))
    assert resp.status_code == 200

    resp = client.post(
        "/pause_generation",
        json={},
        headers=admin_headers(),
    )
    assert resp.status_code == 200


def test_admin_routes_env_key_is_used_when_no_explicit_key(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ADMIN_KEY", "env-key")
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="voicing-tts"))

    resp = client.get("/model_info")
    assert resp.status_code == 401

    resp = client.get("/model_info", headers=admin_headers("env-key"))
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Stub endpoint 501 tests
# ---------------------------------------------------------------------------


def test_unimplemented_tensor_weight_update_returns_501() -> None:
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="voicing-tts"))

    resp = client.post("/update_weights_from_tensor", json={})
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "not_implemented"
    assert "update_weights_from_disk" in resp.json()["error"]["message"]


def test_distributed_weight_update_routes_forward_to_client() -> None:
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="voicing-tts"))

    init = client.post(
        "/init_weights_update_group",
        json={
            "master_address": "10.0.0.1",
            "master_port": 12355,
            "world_size": 2,
            "rank_offset": 1,
            "stages": ["talker"],
            "timeout_s": 0,
        },
    )
    update = client.post(
        "/update_weights_from_distributed",
        json={
            "names": ["w.0"],
            "dtypes": ["bfloat16"],
            "shapes": [[2, 2]],
            "group_name": "weight_update_group",
            "weight_version": "v2",
            "timeout_s": 0,
        },
    )
    destroy = client.post(
        "/destroy_weights_update_group",
        json={
            "group_name": "weight_update_group",
            "stages": ["talker"],
            "timeout_s": 0,
        },
    )

    assert init.status_code == 200
    assert update.status_code == 200
    assert destroy.status_code == 200
    assert admin.calls == [
        (
            "init_weights_update_group",
            {
                "master_address": "10.0.0.1",
                "master_port": 12355,
                "world_size": 2,
                "rank_offset": 1,
                "group_name": "weight_update_group",
                "backend": "nccl",
            },
            ["talker"],
            0,
        ),
        (
            "update_weights_from_distributed",
            {
                "names": ["w.0"],
                "dtypes": ["bfloat16"],
                "shapes": [[2, 2]],
                "group_name": "weight_update_group",
                "flush_cache": True,
                "abort_all_requests": False,
                "weight_version": "v2",
                "torch_empty_cache": False,
            },
            None,
            0,
        ),
        (
            "destroy_weights_update_group",
            {"group_name": "weight_update_group"},
            ["talker"],
            0,
        ),
    ]


def test_stub_endpoint_checks_auth_before_501() -> None:
    """Auth check fires before the tensor stub 501 body."""
    admin = AdminClient()
    client = TestClient(
        create_app(admin, model_name="voicing-tts", admin_api_key=ADMIN_API_KEY)
    )

    resp = client.post("/update_weights_from_tensor", json={})
    assert resp.status_code == 401


@pytest.mark.parametrize("stream", [False, True])
def test_speech_empty_generation_error_allows_next_request(stream: bool) -> None:
    message = "Voicing-TTS generated no audio frames. Please retry the request."

    class FailOnceClient(SuccessfulSpeechClient):
        failed = False

        async def generate(self, request, request_id=None):
            if not self.failed:
                self.failed = True
                raise RuntimeError(message)
            async for chunk in super().generate(request, request_id):
                yield chunk

        async def speech(self, request, **kwargs):
            if not self.failed:
                self.failed = True
                raise ClientError(message)
            return await super().speech(request, **kwargs)

        async def abort(self, request_id):
            pass

    client = TestClient(create_app(FailOnceClient(), model_name="voicing-tts"))
    body = {
        "model": "voicing-tts",
        "input": "hello",
        "stream": stream,
        "response_format": "pcm" if stream else "wav",
    }
    response = client.post("/v1/audio/speech", json=body)
    assert response.status_code == 500
    assert message in response.json()["error"]["message"]
    response = client.post("/v1/audio/speech", json=body)
    assert response.status_code == 200
    assert response.content
