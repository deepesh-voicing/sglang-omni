# SPDX-License-Identifier: Apache-2.0
"""Client wrapper for coordinator-based pipelines."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import aclosing
from dataclasses import replace
from typing import Any, AsyncIterator, Callable

import numpy as np

from sglang_omni.client.audio import FORMAT_MIME_TYPES, encode_audio, to_numpy
from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    ClientError,
    GenerateChunk,
    GenerateRequest,
    SpeechResult,
    UsageInfo,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, RequestState, StreamMessage
from sglang_omni.proto.request import EXPLICIT_STAGE_SAMPLING_PARAMS_KEY


class Client:
    """Internal client used by API adapters."""

    def __init__(
        self,
        coordinator: Coordinator,
        result_builder: Callable[[str, Any], GenerateChunk] | None = None,
        stream_builder: Callable[[str, StreamMessage], GenerateChunk] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.result_builder = result_builder or self.default_result_builder
        self.stream_builder = stream_builder or self.default_stream_builder

    # ------------------------------------------------------------------
    # Low-level generate (backward compatible)
    # ------------------------------------------------------------------

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        req_id = request_id or str(uuid.uuid4())
        omni_request = self.build_omni_request(request)
        if request.stream:
            coordinator_stream = self.coordinator.stream(req_id, omni_request)
            async with aclosing(coordinator_stream):
                async for msg in coordinator_stream:
                    if isinstance(msg, StreamMessage):
                        yield self.stream_builder(req_id, msg)
                    else:
                        yield self.result_builder(req_id, msg.result)
            return
        else:
            pass

        result = await self.coordinator.submit(req_id, omni_request)
        yield self.result_builder(req_id, result)

    # ------------------------------------------------------------------
    # High-level: non-streaming completion
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # High-level: streaming completion
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # High-level: text-to-speech
    # ------------------------------------------------------------------

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ) -> SpeechResult:
        """Run a TTS request and return encoded audio bytes.

        Raises:
            ClientError: If the pipeline produces no audio output.
        """
        audio_chunks: list[Any] = []
        sample_rate: int | None = None
        last_chunk: GenerateChunk | None = None
        extra_params = dict(request.extra_params)
        extra_params.pop("stream", None)
        request = replace(request, stream=False, extra_params=extra_params)

        async for chunk in self.generate(request, request_id=request_id):
            if chunk.audio_data is not None:
                audio_chunks.append(chunk.audio_data)
            else:
                pass
            if chunk.sample_rate is not None:
                sample_rate = chunk.sample_rate
            else:
                pass
            last_chunk = chunk

        if not audio_chunks:
            raise ClientError("No audio output generated from the pipeline.")
        else:
            pass

        if len(audio_chunks) == 1:
            audio_data = audio_chunks[0]
        else:
            arrays = [to_numpy(c) for c in audio_chunks]
            axis = -1 if arrays[0].ndim > 1 else 0
            audio_data = np.concatenate(arrays, axis=axis)

        encode_kwargs: dict[str, Any] = {
            "response_format": response_format,
            "speed": speed,
            "allow_format_fallback": allow_format_fallback,
        }
        if sample_rate is not None:
            encode_kwargs["sample_rate"] = sample_rate
        else:
            pass

        audio_bytes, mime_type = await asyncio.to_thread(
            encode_audio, audio_data, **encode_kwargs
        )

        # Derive actual format from MIME type (encode_audio may fall back
        # to WAV if the requested codec is unavailable).
        actual_format = response_format
        for ext, mt in FORMAT_MIME_TYPES.items():
            if mt == mime_type:
                actual_format = ext
                break
            else:
                pass

        return SpeechResult(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            format=actual_format,
            sample_rate=sample_rate,
            usage=last_chunk.usage if last_chunk else None,
            finish_reason=last_chunk.finish_reason if last_chunk else None,
        )

    # ------------------------------------------------------------------
    # Other operations
    # ------------------------------------------------------------------

    async def abort(
        self,
        request_id: str,
        level: AbortLevel = AbortLevel.SOFT,
    ) -> AbortResult:
        success = await self.coordinator.abort(request_id)
        return AbortResult(success=success, level_applied=level)

    async def get_status(self, request_id: str) -> RequestState | None:
        info = self.coordinator.get_request_info(request_id)
        if info is None:
            return None
        else:
            pass
        return info.state

    def health(self) -> dict[str, Any]:
        return self.coordinator.health()

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.coordinator.admin(
            action,
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def model_info(
        self,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return await self.coordinator.model_info(
            stages=stages,
            timeout_s=timeout_s,
        )

    async def pause_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.coordinator.pause_generation(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def continue_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.coordinator.continue_generation(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_disk(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.coordinator.update_weights_from_disk(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def init_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.coordinator.init_weights_update_group(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def destroy_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.coordinator.destroy_weights_update_group(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_distributed(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.coordinator.update_weights_from_distributed(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def weights_checker(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.coordinator.weights_checker(
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def set_audio_data(chunk: GenerateChunk, data: dict[str, Any]) -> None:
        audio_data = data.get("audio_data") or data.get("audio")
        if audio_data is None and data.get("audio_waveform") is not None:
            raw = data.get("audio_waveform")
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            else:
                pass
            dtype = np.dtype(data.get("audio_waveform_dtype", "float32"))
            arr = np.frombuffer(raw, dtype=dtype)
            shape = data.get("audio_waveform_shape")
            if shape:
                arr = arr.reshape(shape)
            else:
                pass
            audio_data = arr.copy()
        else:
            pass
        if audio_data is not None:
            chunk.audio_data = audio_data
            chunk.modality = "audio"
        else:
            pass
        sample_rate = data.get("sample_rate")
        if sample_rate is not None:
            chunk.sample_rate = sample_rate
        else:
            pass

    @staticmethod
    def build_usage_info(data: dict[str, Any]) -> UsageInfo | None:
        usage = dict(data.get("usage") or {})
        if "prompt_tokens" not in usage and data.get("prompt_tokens") is not None:
            usage["prompt_tokens"] = data.get("prompt_tokens")
        else:
            pass
        if (
            "completion_tokens" not in usage
            and data.get("completion_tokens") is not None
        ):
            usage["completion_tokens"] = data.get("completion_tokens")
        else:
            pass
        if "total_tokens" not in usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if prompt_tokens is not None or completion_tokens is not None:
                usage["total_tokens"] = (prompt_tokens or 0) + (completion_tokens or 0)
            else:
                pass
        else:
            pass
        if "engine_time_s" not in usage and data.get("engine_time_s") is not None:
            usage["engine_time_s"] = data.get("engine_time_s")
        else:
            pass
        return UsageInfo.from_dict(usage)

    @staticmethod
    def build_omni_request(request: GenerateRequest) -> OmniRequest:
        inputs = extract_inputs(request)
        params = build_params(request)
        metadata = dict(request.metadata)
        if (
            request.stage_sampling
            and EXPLICIT_STAGE_SAMPLING_PARAMS_KEY not in metadata
        ):
            metadata[EXPLICIT_STAGE_SAMPLING_PARAMS_KEY] = {
                stage: list(sampling)
                for stage, sampling in params["stage_sampling"].items()
            }
        else:
            pass
        if request.model:
            metadata.setdefault("model", request.model)
        else:
            pass
        if request.output_modalities:
            metadata["output_modalities"] = request.output_modalities
        else:
            pass
        return OmniRequest(inputs=inputs, params=params, metadata=metadata)

    @staticmethod
    def default_result_builder(request_id: str, result: Any) -> GenerateChunk:
        chunk = GenerateChunk(request_id=request_id, finish_reason="stop")
        if isinstance(result, GenerateChunk):
            result.request_id = request_id
            return result
        else:
            pass
        if isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str):
                chunk.text = text
            else:
                pass
            token_ids = result.get("token_ids")
            if token_ids is not None:
                if not isinstance(token_ids, (list, tuple)):
                    token_ids = token_ids.tolist()
                else:
                    pass
                chunk.token_ids = list(token_ids)
            else:
                pass
            logprobs = result.get("logprobs")
            if logprobs is not None:
                chunk.logprobs = logprobs
            else:
                pass
            output_token_logprobs = result.get("output_token_logprobs")
            if output_token_logprobs is not None:
                chunk.output_token_logprobs = output_token_logprobs
            else:
                pass
            omni_rollout = result.get("omni_rollout")
            if omni_rollout is not None:
                chunk.omni_rollout = omni_rollout
            else:
                pass
            weight_version = result.get("weight_version")
            if weight_version is not None:
                chunk.weight_version = weight_version
            else:
                pass
            finish_reason = result.get("finish_reason")
            if finish_reason is not None:
                chunk.finish_reason = finish_reason
            else:
                pass
            chunk.stage_id = result.get("stage_id")
            chunk.stage_name = result.get("stage_name")
            modality = result.get("modality")
            if modality is not None:
                chunk.modality = modality
            else:
                pass
            language = result.get("language")
            if isinstance(language, str):
                chunk.language = language
            else:
                pass
            Client.set_audio_data(chunk, result)
            chunk.usage = Client.build_usage_info(result)
            return chunk
        else:
            pass
        if isinstance(result, str):
            chunk.text = result
            return chunk
        else:
            pass
        chunk.text = str(result)
        return chunk

    @staticmethod
    def default_stream_builder(request_id: str, msg: StreamMessage) -> GenerateChunk:
        chunk = GenerateChunk(request_id=request_id)
        chunk.stage_name = msg.stage_name or msg.from_stage
        chunk.stage_id = msg.stage_id
        if msg.modality:
            chunk.modality = msg.modality
        else:
            pass

        data = msg.chunk
        if isinstance(data, GenerateChunk):
            data.request_id = request_id
            if data.stage_name is None:
                data.stage_name = chunk.stage_name
            else:
                pass
            if data.stage_id is None:
                data.stage_id = chunk.stage_id
            else:
                pass
            if not data.modality and chunk.modality:
                data.modality = chunk.modality
            else:
                pass
            return data
        else:
            pass
        if isinstance(data, dict):
            text = data.get("text")
            if isinstance(text, str):
                chunk.text = text
            else:
                pass
            token_ids = data.get("token_ids")
            if token_ids is not None:
                if not isinstance(token_ids, (list, tuple)):
                    token_ids = token_ids.tolist()
                else:
                    pass
                chunk.token_ids = list(token_ids)
            else:
                pass
            logprobs = data.get("logprobs")
            if logprobs is not None:
                chunk.logprobs = logprobs
            else:
                pass
            output_token_logprobs = data.get("output_token_logprobs")
            if output_token_logprobs is not None:
                chunk.output_token_logprobs = output_token_logprobs
            else:
                pass
            omni_rollout = data.get("omni_rollout")
            if omni_rollout is not None:
                chunk.omni_rollout = omni_rollout
            else:
                pass
            weight_version = data.get("weight_version")
            if weight_version is not None:
                chunk.weight_version = weight_version
            else:
                pass
            finish_reason = data.get("finish_reason")
            if finish_reason is not None:
                chunk.finish_reason = finish_reason
            else:
                pass
            chunk.usage = Client.build_usage_info(data)
            stage_name = data.get("stage_name")
            if stage_name is not None:
                chunk.stage_name = stage_name
            else:
                pass
            stage_id = data.get("stage_id")
            if stage_id is not None:
                chunk.stage_id = stage_id
            else:
                pass
            modality = data.get("modality")
            if modality is not None:
                chunk.modality = modality
            else:
                pass
            Client.set_audio_data(chunk, data)
            return chunk
        else:
            pass
        if isinstance(data, str):
            chunk.text = data
            return chunk
        else:
            pass
        if isinstance(data, int):
            chunk.token_ids = [data]
            return chunk
        else:
            pass
        chunk.text = str(data)
        return chunk


def extract_inputs(request: GenerateRequest) -> Any:
    if request.prompt is None:
        raise ValueError("GenerateRequest requires a prompt.")
    else:
        pass
    return request.prompt


def build_params(request: GenerateRequest) -> dict[str, Any]:
    params = request.sampling.to_dict()
    max_new_tokens = request.sampling.max_new_tokens
    if request.max_tokens is not None:
        max_new_tokens = request.max_tokens
    else:
        pass
    if max_new_tokens is None:
        params.pop("max_new_tokens", None)
    else:
        params["max_new_tokens"] = max_new_tokens
    params["stream"] = request.stream
    if request.stage_sampling:
        params["stage_sampling"] = {
            key: value.to_dict() for key, value in request.stage_sampling.items()
        }
    else:
        pass
    if request.stage_params:
        params["stage_params"] = request.stage_params
    else:
        pass
    if request.extra_params:
        params.update(request.extra_params)
    else:
        pass
    return params
