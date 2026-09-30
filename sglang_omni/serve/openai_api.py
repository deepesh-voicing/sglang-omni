# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for sglang-omni.

Provides the following endpoints:
- POST /v1/audio/speech      — Text-to-speech synthesis
- POST /v1/audio/speech/batch — Batch text-to-speech synthesis
- WS   /v1/audio/speech/stream — Stateful TTS WebSocket streaming
- GET  /v1/audio/voices      — List preset and uploaded TTS voices
- POST /v1/audio/voices      — Upload a persistent TTS reference voice
- DELETE /v1/audio/voices/{name} — Delete an uploaded TTS voice
- GET  /v1/models            — List available models
- GET  /health               — Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, AsyncIterator

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni import __version__
from sglang_omni.client import Client, ClientError, GenerateRequest
from sglang_omni.client.audio import (
    DEFAULT_SAMPLE_RATE,
    apply_speed,
    encode_pcm,
    select_audio_delta,
)
from sglang_omni.config import CustomVoiceConfig
from sglang_omni.config.schema import MAX_SPEECH_INPUT_CHARS
from sglang_omni.http.admin_auth import (
    make_admin_auth_dependency,
    resolve_admin_api_key,
)
from sglang_omni.http.favicon import register_favicon
from sglang_omni.serve.protocol import (
    DEFAULT_TTS_BATCH_MAX_ITEMS,
    AdminRequestBase,
    ContinueGenerationRequest,
    CreateSpeechBatchRequest,
    DestroyWeightsUpdateGroupRequest,
    InitWeightsUpdateGroupRequest,
    ModelCard,
    ModelList,
    PauseGenerationRequest,
    SpeechBatchResponse,
    UpdateWeightFromDiskRequest,
    UpdateWeightsFromDistributedRequest,
    VoiceListResponse,
    WeightsCheckerRequest,
)
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    openai_error_payload,
    speech_error_response,
    speech_generation_error,
)
from sglang_omni.serve.speech_limits import (
    MAX_VOICE_UPLOAD_BODY_BYTES,
    MAX_VOICE_UPLOAD_BYTES,
)
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.serve.speech_voices import SpeakerSampleStore
from sglang_omni.serve.speech_ws import SpeechWebSocketSession
from sglang_omni.serve.streaming import (
    close_async_iterator_if_supported as _close_async_iterator_if_supported,
)

logger = logging.getLogger(__name__)
HTTP_DISCONNECT_POLL_INTERVAL_S = 0.05
HTTP_DISCONNECT_CANCEL_TIMEOUT_S = 0.1


class RequestBodyTooLarge(Exception):
    pass


class VoiceUploadBodyLimitMiddleware:
    """Reject oversized voice uploads before Starlette parses multipart bodies."""

    def __init__(self, app: Callable[..., Awaitable[None]], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if not is_voice_upload_scope(scope):
            await self.app(scope, receive, send)
            return
        else:
            pass

        request_content_length = content_length(scope)
        if (
            request_content_length is not None
            and request_content_length > self.max_bytes
        ):
            await send_voice_upload_too_large(send, self.max_bytes)
            return
        else:
            pass

        received_bytes = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    raise RequestBodyTooLarge
                else:
                    pass
            else:
                pass
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await send_voice_upload_too_large(send, self.max_bytes)


def create_app(
    client: Client,
    *,
    model_name: str | None = None,
    requires_uploaded_voice_for_named_voice: bool = False,
    supports_uploaded_voice_references: bool = True,
    custom_voice_config: CustomVoiceConfig | None = None,
    required_speech_reference_count: int | None = None,
    speech_reference_text_required: bool = False,
    speech_reference_text_excludes_instructions: bool = False,
    additional_speech_languages: frozenset[str] = frozenset(),
    max_speech_input_chars: int | None = MAX_SPEECH_INPUT_CHARS,
    allowed_local_media_path: str | None = None,
    allowed_media_domains: list[str] | None = None,
    admin_api_key: str | None = None,
    tts_batch_max_items: int = DEFAULT_TTS_BATCH_MAX_ITEMS,
    architectures: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        client: Client instance connected to the pipeline coordinator.
        model_name: Default model name to report in responses and /v1/models.
        requires_uploaded_voice_for_named_voice: Whether non-default TTS voice
            names must resolve to uploaded voices before reaching the model.
        supports_uploaded_voice_references: Whether uploaded voice names can be
            lowered into backend reference-audio requests.
        custom_voice_config: Checkpoint speaker names and task type for CustomVoice.
            When present, reference inputs and uploaded-voice resolution are disabled.
        required_speech_reference_count: Exact reference count required before
            dispatching a speech request to the backend.
        speech_reference_text_required: Whether each speech reference requires
            a transcript.
        speech_reference_text_excludes_instructions: Whether a reference
            transcript and style instructions are mutually exclusive.
        additional_speech_languages: Pipeline-specific accepted languages.
        max_speech_input_chars: Maximum accepted input characters, or ``None``
            to defer length validation to model-specific context checks.
        allowed_local_media_path: Directory that local media references in TTS
            requests must resolve inside. ``file://`` references are disabled
            when omitted; bare local paths remain allowed by default but are
            also restricted to this directory once it is configured.
        allowed_media_domains: Domains allowed for remote TTS reference audio.
        admin_api_key: Optional API key for admin-control endpoints.
        tts_batch_max_items: Maximum items accepted by
            ``/v1/audio/speech/batch``.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="sglang-omni", version=__version__)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        VoiceUploadBodyLimitMiddleware,
        max_bytes=MAX_VOICE_UPLOAD_BODY_BYTES,
    )

    # Store references in app state for access from route handlers
    app.state.client = client
    app.state.model_name = model_name or "sglang-omni"
    app.state.architectures = [a for a in (architectures or []) if a]
    app.state.speaker_sample_store = SpeakerSampleStore()
    app.state.speech_service = SpeechRequestValidator(
        default_model=app.state.model_name,
        custom_voice_config=custom_voice_config,
        requires_uploaded_voice_for_named_voice=(
            requires_uploaded_voice_for_named_voice
        ),
        supports_uploaded_voice_references=supports_uploaded_voice_references,
        required_speech_reference_count=required_speech_reference_count,
        speech_reference_text_required=speech_reference_text_required,
        speech_reference_text_excludes_instructions=(
            speech_reference_text_excludes_instructions
        ),
        additional_speech_languages=additional_speech_languages,
        max_speech_input_chars=max_speech_input_chars,
        allowed_local_media_path=allowed_local_media_path,
        allowed_media_domains=allowed_media_domains,
        voice_store=app.state.speaker_sample_store,
        tts_batch_max_items=tts_batch_max_items,
    )

    resolved_key = resolve_admin_api_key(admin_api_key)

    # Register all routes
    register_favicon(app)
    register_health(app)
    register_models(app)
    register_admin(app, resolved_key)
    register_voices(app)
    register_speech(app)
    register_speech_batch(app)
    register_speech_ws(app)

    return app


def register_voices(app: FastAPI) -> None:
    @app.get("/v1/audio/voices")
    async def list_voices(names_only: bool = False) -> JSONResponse:
        voice_store: SpeakerSampleStore = app.state.speaker_sample_store
        if names_only:
            return JSONResponse(
                content={"uploaded_voice_names": voice_store.uploaded_voice_names()}
            )
        else:
            pass
        voice_list = voice_store.list_response()
        custom_voice_config = app.state.speech_service.custom_voice_config
        if custom_voice_config is not None:
            voices = {name.casefold(): name for name in custom_voice_config.speakers}
            voices["default"] = "default"
            voice_list["voices"] = sorted(voices.values(), key=str.casefold)
        else:
            pass
        response = VoiceListResponse.model_validate(voice_list)
        return JSONResponse(content=response.model_dump(exclude_none=True))

    @app.post("/v1/audio/voices")
    async def upload_voice(
        audio_sample: UploadFile = File(...),
        consent: str = Form(...),
        name: str = Form(...),
        ref_text: str | None = Form(default=None),
        speaker_description: str | None = Form(default=None),
    ) -> JSONResponse:
        voice_store: SpeakerSampleStore = app.state.speaker_sample_store
        try:
            response = voice_store.upload(
                name=name,
                consent=consent,
                audio_bytes=await read_voice_upload(audio_sample),
                filename=audio_sample.filename,
                content_type=audio_sample.content_type,
                ref_text=ref_text,
                speaker_description=speaker_description,
            )
        except SpeechAPIError as exc:
            return speech_error_response(exc)
        return JSONResponse(content=response)

    @app.delete("/v1/audio/voices/{name}")
    async def delete_voice(name: str) -> JSONResponse:
        voice_store: SpeakerSampleStore = app.state.speaker_sample_store
        try:
            deleted = voice_store.delete(name)
        except SpeechAPIError as exc:
            return speech_error_response(exc)
        if not deleted:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Voice '{name}' not found"},
            )
        else:
            pass
        return JSONResponse(
            content={
                "success": True,
                "message": f"Voice '{name}' deleted successfully",
            }
        )


async def read_voice_upload(audio_sample: UploadFile) -> bytes:
    audio_bytes = await audio_sample.read(MAX_VOICE_UPLOAD_BYTES + 1)
    if len(audio_bytes) > MAX_VOICE_UPLOAD_BYTES:
        raise bad_request(
            f"audio_sample must be at most {MAX_VOICE_UPLOAD_BYTES} bytes",
            param="audio_sample",
        )
    else:
        pass
    return audio_bytes


def is_voice_upload_scope(scope: dict[str, Any]) -> bool:
    return (
        scope.get("type") == "http"
        and scope.get("method") == "POST"
        and scope.get("path") == "/v1/audio/voices"
    )


def content_length(scope: dict[str, Any]) -> int | None:
    for name, value in scope.get("headers", ()):
        if name.lower() != b"content-length":
            continue
        else:
            pass
        try:
            return int(value.decode("ascii"))
        except ValueError:
            return None
    return None


async def send_voice_upload_too_large(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    max_bytes: int,
) -> None:
    body = json.dumps(
        openai_error_payload(
            f"request body must be at most {max_bytes} bytes",
            error_type="RequestTooLargeError",
            param="audio_sample",
            code=413,
        )
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def register_health(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint (includes filesystem browse info)."""
        client: Client = app.state.client
        info = client.health()
        is_running = info.get("running", False)
        status_code = 200 if is_running else 503
        return JSONResponse(
            content={
                "status": "healthy" if is_running else "unhealthy",
                **info,
            },
            status_code=status_code,
        )


def register_models(app: FastAPI) -> None:
    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """List available models."""
        model_name: str = app.state.model_name
        model_list = ModelList(
            data=[
                ModelCard(
                    id=model_name,
                    root=model_name,
                    created=0,
                )
            ]
        )
        return JSONResponse(content=model_list.model_dump())


def register_admin(app: FastAPI, admin_api_key: str | None = None) -> None:
    _auth = make_admin_auth_dependency(admin_api_key)

    @app.get("/model_info", dependencies=[Depends(_auth)])
    async def model_info_get() -> JSONResponse:
        client: Client = app.state.client
        return model_info_response(await client.model_info())

    @app.post("/model_info", dependencies=[Depends(_auth)])
    async def model_info_post(req: AdminRequestBase) -> JSONResponse:
        client: Client = app.state.client
        return model_info_response(
            await client.model_info(
                stages=req.stages,
                timeout_s=timeout_or_default(req.timeout_s, 30.0),
            )
        )

    @app.post("/pause_generation", dependencies=[Depends(_auth)])
    async def pause_generation(req: PauseGenerationRequest) -> JSONResponse:
        client: Client = app.state.client
        payload = request_payload(req)
        return admin_response(
            await client.pause_generation(
                payload,
                stages=req.stages,
                timeout_s=timeout_or_default(req.timeout_s, 60.0),
            )
        )

    @app.post("/continue_generation", dependencies=[Depends(_auth)])
    async def continue_generation(req: ContinueGenerationRequest) -> JSONResponse:
        client: Client = app.state.client
        payload = request_payload(req)
        return admin_response(
            await client.continue_generation(
                payload,
                stages=req.stages,
                timeout_s=timeout_or_default(req.timeout_s, 60.0),
            )
        )

    @app.post("/update_weights_from_disk", dependencies=[Depends(_auth)])
    async def update_weights_from_disk(
        req: UpdateWeightFromDiskRequest,
    ) -> JSONResponse:
        client: Client = app.state.client
        payload = request_payload(req)
        return admin_response(
            await client.update_weights_from_disk(
                payload,
                stages=req.stages,
                timeout_s=timeout_or_default(req.timeout_s, 120.0),
            )
        )

    @app.post("/update_weights_from_tensor", dependencies=[Depends(_auth)])
    async def update_weights_from_tensor(
        request: Request,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": (
                        "update_weights_from_tensor is not yet implemented. "
                        "Use update_weights_from_disk for the disk-based weight update path."
                    ),
                    "code": "not_implemented",
                }
            },
        )

    @app.post("/init_weights_update_group", dependencies=[Depends(_auth)])
    async def init_weights_update_group(
        req: InitWeightsUpdateGroupRequest,
    ) -> JSONResponse:
        client: Client = app.state.client
        payload = request_payload(req)
        return admin_response(
            await client.init_weights_update_group(
                payload,
                stages=req.stages,
                timeout_s=timeout_or_default(req.timeout_s, 300.0),
            )
        )

    @app.post("/destroy_weights_update_group", dependencies=[Depends(_auth)])
    async def destroy_weights_update_group(
        req: DestroyWeightsUpdateGroupRequest,
    ) -> JSONResponse:
        client: Client = app.state.client
        payload = request_payload(req)
        return admin_response(
            await client.destroy_weights_update_group(
                payload,
                stages=req.stages,
                timeout_s=timeout_or_default(req.timeout_s, 300.0),
            )
        )

    @app.post("/update_weights_from_distributed", dependencies=[Depends(_auth)])
    async def update_weights_from_distributed(
        req: UpdateWeightsFromDistributedRequest,
    ) -> JSONResponse:
        client: Client = app.state.client
        payload = request_payload(req)
        return admin_response(
            await client.update_weights_from_distributed(
                payload,
                stages=req.stages,
                timeout_s=timeout_or_default(req.timeout_s, 300.0),
            )
        )

    @app.get("/weights_checker", dependencies=[Depends(_auth)])
    async def weights_checker_get(action: str = "checksum") -> JSONResponse:
        client: Client = app.state.client
        return admin_response(await client.weights_checker({"action": action}))

    @app.post("/weights_checker", dependencies=[Depends(_auth)])
    async def weights_checker_post(req: WeightsCheckerRequest) -> JSONResponse:
        client: Client = app.state.client
        payload = request_payload(req)
        return admin_response(
            await client.weights_checker(
                payload,
                stages=req.stages,
                timeout_s=timeout_or_default(req.timeout_s, 120.0),
            )
        )


def timeout_or_default(timeout_s: float | None, default: float) -> float:
    return default if timeout_s is None else timeout_s


def request_payload(req: AdminRequestBase) -> dict[str, Any]:
    return req.model_dump(exclude={"stages", "timeout_s"}, exclude_none=True)


def admin_response(result: dict[str, Any]) -> JSONResponse:
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result)
    else:
        pass
    return JSONResponse(content=result)


def model_info_response(result: dict[str, Any]) -> JSONResponse:
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result)
    else:
        pass

    stage_infos = extract_model_info_stage_data(result)
    weight_version = common_model_info_value(
        result,
        stage_infos,
        "weight_version",
        mixed_status_code=409,
    )
    payload = dict(result)
    payload.update(
        {
            "weight_version": weight_version,
            "model_path": common_model_info_value(result, stage_infos, "model_path"),
            "load_format": common_model_info_value(result, stage_infos, "load_format"),
            "stages": result.get("results", []),
        }
    )
    return JSONResponse(content=payload)


def extract_model_info_stage_data(result: dict[str, Any]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for item in result.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        else:
            pass
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        else:
            pass
        if data.get("skipped") or data.get("unsupported"):
            continue
        else:
            pass
        stage_info = dict(data)
        stage_info.setdefault("stage", item.get("stage"))
        stage_info.setdefault("success", item.get("success"))
        infos.append(stage_info)
    return infos


def common_model_info_value(
    result: dict[str, Any],
    stage_infos: list[dict[str, Any]],
    key: str,
    *,
    mixed_status_code: int | None = None,
) -> Any:
    values = [info[key] for info in stage_infos if info.get(key) is not None]
    if not values:
        return None
    else:
        pass

    unique: dict[str, Any] = {}
    for value in values:
        unique.setdefault(json.dumps(value, sort_keys=True, default=str), value)
    if len(unique) == 1:
        return next(iter(unique.values()))
    else:
        pass
    if mixed_status_code is not None:
        raise HTTPException(
            status_code=mixed_status_code,
            detail={
                "success": False,
                "message": f"mixed stage {key}",
                "mixed_state": {key: list(unique.values())},
                "stages": stage_infos,
                "admin": result,
            },
        )
    else:
        pass
    return None


def speech_generation_failure_response(
    request_id: str,
    exc: BaseException,
    *,
    unexpected_message: str | None = None,
) -> JSONResponse:
    mapped = speech_generation_error(exc)
    if mapped.status_code not in (400, 503):
        logger.exception(
            unexpected_message or "Error generating speech for request %s",
            request_id,
        )
    else:
        logger.warning(
            "Rejecting speech request %s: %s",
            request_id,
            mapped.message,
        )
    return speech_error_response(mapped)


def register_speech(app: FastAPI) -> None:
    @app.post("/v1/audio/speech")
    async def create_speech(request: Request) -> Response:
        client: Client = app.state.client
        speech_service: SpeechRequestValidator = app.state.speech_service

        request_id = f"speech-{uuid.uuid4()}"
        try:
            payload = await request.json()
            prepared = await asyncio.to_thread(
                speech_service.parse_generation_request, payload
            )
            req = prepared.request
            gen_req = speech_service.build_generate_request(
                req,
                validate=False,
                reference_descriptors=prepared.reference_descriptors,
                uploaded_voice=prepared.uploaded_voice,
            )
        except json.JSONDecodeError:
            return speech_error_response(
                bad_request("speech request body must be valid JSON")
            )
        except SpeechAPIError as exc:
            return speech_error_response(exc)

        if req.stream:
            try:
                return await speech_audio_response(
                    request=request,
                    client=client,
                    gen_req=gen_req,
                    request_id=request_id,
                    speed=req.speed,
                )
            except ClientError as exc:
                return speech_generation_failure_response(request_id, exc)
            except Exception as exc:
                return speech_generation_failure_response(
                    request_id,
                    exc,
                    unexpected_message=(
                        "Error preparing raw PCM speech stream for request %s"
                    ),
                )
        else:
            pass

        try:
            result = await await_speech_response(
                request=request,
                client=client,
                gen_req=gen_req,
                request_id=request_id,
                response_format=req.response_format,
                speed=req.speed,
            )
        except ClientError as exc:
            return speech_generation_failure_response(request_id, exc)
        except Exception as exc:
            return speech_generation_failure_response(
                request_id,
                exc,
                unexpected_message="Error generating speech for request %s",
            )

        headers = {
            "Content-Disposition": f'attachment; filename="speech.{result.format}"',
        }
        if result.finish_reason is not None:
            # note (Junnan Li): the body is binary audio, so the terminal state
            # travels in the same X- header channel as usage.
            headers["X-Finish-Reason"] = str(result.finish_reason)
        else:
            pass
        if result.usage is not None:
            if result.usage.prompt_tokens is not None:
                headers["X-Prompt-Tokens"] = str(result.usage.prompt_tokens)
            else:
                pass
            if result.usage.completion_tokens is not None:
                headers["X-Completion-Tokens"] = str(result.usage.completion_tokens)
            else:
                pass
            if result.usage.engine_time_s is not None:
                headers["X-Engine-Time"] = str(result.usage.engine_time_s)
            else:
                pass
        else:
            pass

        return Response(
            content=result.audio_bytes,
            media_type=result.mime_type,
            headers=headers,
        )


def register_speech_batch(app: FastAPI) -> None:
    @app.post("/v1/audio/speech/batch")
    async def create_speech_batch(request: Request) -> JSONResponse:
        client: Client = app.state.client
        speech_service: SpeechRequestValidator = app.state.speech_service
        request_id = f"speech-batch-{uuid.uuid4()}"
        try:
            payload = await request.json()
            batch = await asyncio.to_thread(speech_service.parse_batch_request, payload)
            response = await create_speech_batch_with_disconnect_watch(
                request,
                client=client,
                speech_service=speech_service,
                batch=batch,
                request_id=request_id,
            )
        except json.JSONDecodeError:
            return speech_error_response(
                bad_request("speech batch request body must be valid JSON")
            )
        except SpeechAPIError as exc:
            return speech_error_response(exc)
        except Exception as exc:
            mapped = speech_generation_error(exc)
            if mapped.status_code not in (400, 503):
                logger.exception(
                    "Error generating speech batch for request %s", request_id
                )
            else:
                logger.warning(
                    "Rejecting speech batch request %s: %s",
                    request_id,
                    mapped.message,
                )
            return speech_error_response(mapped)

        response = SpeechBatchResponse.model_validate(response)
        return JSONResponse(content=response.model_dump(exclude_none=True))


async def create_speech_batch_with_disconnect_watch(
    request: Request,
    *,
    client: Client,
    speech_service: SpeechRequestValidator,
    batch: CreateSpeechBatchRequest,
    request_id: str,
) -> Any:
    batch_task = asyncio.create_task(
        speech_service.create_speech_batch(
            client,
            batch,
            request_id=request_id,
        )
    )
    disconnect_task = asyncio.create_task(wait_for_request_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            {batch_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if batch_task in done:
            return await batch_task
        else:
            pass

        batch_task.cancel()
        with suppress(asyncio.CancelledError):
            await batch_task
        raise asyncio.CancelledError
    finally:
        if not disconnect_task.done():
            await cancel_task_bounded(disconnect_task)
        else:
            pass


def register_speech_ws(app: FastAPI) -> None:
    @app.websocket("/v1/audio/speech/stream")
    async def speech_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        session = SpeechWebSocketSession(
            websocket,
            client=app.state.client,
            speech_service=app.state.speech_service,
        )
        await session.run()


def speech_pcm_chunk_bytes(
    chunk: Any,
    *,
    emitted_samples: int,
    speed: float,
) -> tuple[bytes | None, int, int]:
    sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
    audio_data, emitted_samples = select_audio_delta(
        chunk.audio_data,
        emitted_samples=emitted_samples,
        is_terminal=chunk.finish_reason is not None,
    )
    if audio_data is None:
        return None, emitted_samples, sample_rate
    else:
        pass

    if speed != 1.0:
        audio_data, sample_rate = apply_speed(audio_data, speed, sample_rate)
    else:
        pass
    audio_bytes = encode_pcm(audio_data, sample_rate)
    if not audio_bytes:
        return None, emitted_samples, sample_rate
    else:
        pass
    return audio_bytes, emitted_samples, sample_rate


async def speech_audio_response(
    request: Request,
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    speed: float,
) -> StreamingResponse:
    """Build a raw PCM stream after deriving headers from the first audio chunk."""
    emitted_samples = 0
    chunk_stream = client.generate(gen_req, request_id=request_id)
    first_audio_bytes: bytes | None = None
    stream_sample_rate: int | None = None
    stream_completed = False
    stream_closed = False
    disconnect_task = asyncio.create_task(wait_for_request_disconnect(request))
    next_chunk_task: asyncio.Task[Any] | None = None

    try:
        while True:
            next_chunk_task = asyncio.create_task(anext(chunk_stream))
            done, _ = await asyncio.wait(
                {next_chunk_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                if not next_chunk_task.done():
                    await cancel_task_bounded(next_chunk_task)
                else:
                    pass
                await abort_and_close_speech_stream(client, request_id, chunk_stream)
                stream_closed = True
                raise asyncio.CancelledError
            else:
                pass

            try:
                chunk = next_chunk_task.result()
            except StopAsyncIteration:
                stream_completed = True
                break
            if chunk.audio_data is None:
                continue
            else:
                pass

            first_audio_bytes, emitted_samples, stream_sample_rate = (
                speech_pcm_chunk_bytes(
                    chunk,
                    emitted_samples=emitted_samples,
                    speed=speed,
                )
            )
            if first_audio_bytes is not None:
                break
            else:
                pass

        if first_audio_bytes is None or stream_sample_rate is None:
            raise RuntimeError("No audio output generated from the pipeline.")
        else:
            pass
    except asyncio.CancelledError:
        if not stream_closed:
            await abort_and_close_speech_stream(client, request_id, chunk_stream)
        else:
            pass
        raise
    except Exception:
        if not stream_completed:
            await abort_and_close_speech_stream(client, request_id, chunk_stream)
        else:
            await _close_async_iterator_if_supported(chunk_stream)
        raise
    finally:
        if next_chunk_task is not None and not next_chunk_task.done():
            await cancel_task_bounded(next_chunk_task)
        else:
            pass
        if not disconnect_task.done():
            await cancel_task_bounded(disconnect_task)
        else:
            pass

    async def _body():
        nonlocal emitted_samples
        active_request = True
        try:
            yield first_audio_bytes

            async for chunk in chunk_stream:
                if chunk.audio_data is None:
                    continue
                else:
                    pass

                audio_bytes, emitted_samples, sample_rate = speech_pcm_chunk_bytes(
                    chunk,
                    emitted_samples=emitted_samples,
                    speed=speed,
                )
                if audio_bytes is None:
                    continue
                else:
                    pass
                if sample_rate != stream_sample_rate:
                    raise RuntimeError(
                        "Raw PCM speech stream sample rate changed from "
                        f"{stream_sample_rate} to {sample_rate}"
                    )
                else:
                    pass
                yield audio_bytes
            active_request = False
        finally:
            if active_request:
                await abort_and_close_speech_stream(client, request_id, chunk_stream)
            else:
                await _close_async_iterator_if_supported(chunk_stream)

    return StreamingResponse(
        _body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(stream_sample_rate),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


async def await_speech_response(
    request: Request,
    client: Client,
    gen_req: GenerateRequest,
    *,
    request_id: str,
    response_format: str,
    speed: float,
):
    speech_task = asyncio.create_task(
        client.speech(
            gen_req,
            request_id=request_id,
            response_format=response_format,
            speed=speed,
            allow_format_fallback=False,
        )
    )
    disconnect_task = asyncio.create_task(wait_for_request_disconnect(request))
    aborted = False
    try:
        done, _ = await asyncio.wait(
            {speech_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if speech_task in done:
            return speech_task.result()
        else:
            pass

        await client.abort(request_id)
        aborted = True
        speech_task.cancel()
        raise asyncio.CancelledError
    except asyncio.CancelledError:
        if not aborted:
            await client.abort(request_id)
        else:
            pass
        raise
    finally:
        if not speech_task.done():
            await cancel_task_bounded(speech_task)
        else:
            pass
        if not disconnect_task.done():
            await cancel_task_bounded(disconnect_task)
        else:
            pass


async def cancel_task_bounded(task: asyncio.Task[Any]) -> None:
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=HTTP_DISCONNECT_CANCEL_TIMEOUT_S)
    if done:
        await asyncio.gather(*done, return_exceptions=True)
    else:
        task.add_done_callback(discard_cancelled_task_result)


def discard_cancelled_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Cancelled request task finished with an error", exc_info=True)


async def wait_for_request_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(HTTP_DISCONNECT_POLL_INTERVAL_S)


async def abort_and_close_speech_stream(
    client: Client,
    request_id: str,
    stream: AsyncIterator[Any],
) -> None:
    try:
        await client.abort(request_id)
    finally:
        await _close_async_iterator_if_supported(stream)
