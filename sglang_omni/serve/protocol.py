# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible request/response protocol definitions."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

SUPPORTED_TTS_RESPONSE_FORMATS = frozenset({"wav", "mp3", "flac", "pcm", "aac", "opus"})
SUPPORTED_TTS_LANGUAGES = frozenset(
    {
        "Auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    }
)
SUPPORTED_TTS_TASK_TYPES = frozenset({"Base", "CustomVoice", "VoiceDesign"})
TTS_SPEED_MIN = 0.25
TTS_SPEED_MAX = 4.0
DEFAULT_TTS_BATCH_MAX_ITEMS = 32


class SpeechReference(BaseModel):
    """Reference item for voice cloning in /v1/audio/speech."""

    audio_path: str | None = None
    ref_audio: str | None = None
    audio: str | None = None
    data: str | None = None
    media_type: str | None = None
    text: str | None = None


class CreateSpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request.

    Standard OpenAI fields plus extensions for advanced TTS models
    (e.g. voice cloning, style instructions).
    """

    model_config = ConfigDict(populate_by_name=True)

    # Standard OpenAI fields
    model: str | None = None
    input: str
    voice: str = Field(
        default="default",
        validation_alias=AliasChoices("voice", "speaker"),
    )
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False

    # Advanced TTS extensions
    task_type: str | None = None  # e.g. "Base", "CustomVoice", "VoiceDesign"
    language: str | None = None
    instructions: str | None = None  # style/emotion instructions

    # Voice cloning parameters
    ref_audio: str | None = None  # path or URL to reference audio
    ref_text: str | None = None  # transcript of reference audio
    references: list[SpeechReference] | None = None
    x_vector_only_mode: bool | None = None
    stream_codec_output: bool | None = None
    suppress_bootstrap_silence: bool | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)

    # Generation parameters
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None

    # Per-stage overrides (sglang-omni specific)
    stage_params: dict[str, dict[str, Any]] | None = None


class SpeechBatchItem(BaseModel):
    """One item in a batch text-to-speech request."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    model: Any = None
    input: Any = None
    voice: Any = Field(
        default=None,
        validation_alias=AliasChoices("voice", "speaker"),
    )
    response_format: Any = None
    speed: Any = None
    stream: Any = None
    task_type: Any = None
    language: Any = None
    instructions: Any = None
    ref_audio: Any = None
    ref_text: Any = None
    references: Any = None
    x_vector_only_mode: Any = None
    stream_codec_output: Any = None
    suppress_bootstrap_silence: Any = None
    max_new_tokens: Any = None
    initial_codec_chunk_frames: Any = None
    temperature: Any = None
    top_p: Any = None
    top_k: Any = None
    repetition_penalty: Any = None
    seed: Any = None
    stage_params: Any = None


class CreateSpeechBatchRequest(BaseModel):
    """Batch text-to-speech request with shared defaults and item overrides."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    items: list[SpeechBatchItem]
    voice: str = Field(
        default="default",
        validation_alias=AliasChoices("voice", "speaker"),
    )
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False
    task_type: str | None = None
    language: str | None = None
    instructions: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    references: list[SpeechReference] | None = None
    x_vector_only_mode: bool | None = None
    stream_codec_output: bool | None = None
    suppress_bootstrap_silence: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    stage_params: dict[str, dict[str, Any]] | None = None


class SpeechBatchResult(BaseModel):
    """One item result in a batch text-to-speech response."""

    index: int
    status: str
    audio_data: str | None = None
    format: str | None = None
    media_type: str | None = None
    finish_reason: str | None = None
    error: dict[str, Any] | None = None


class SpeechBatchResponse(BaseModel):
    """Batch text-to-speech response preserving item order."""

    id: str
    results: list[SpeechBatchResult]
    total: int
    succeeded: int
    failed: int


class SpeechStreamSessionConfig(BaseModel):
    """Configuration for /v1/audio/speech/stream WebSocket sessions."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    voice: str = Field(
        default="default",
        validation_alias=AliasChoices("voice", "speaker"),
    )
    response_format: str = "pcm"
    speed: float = 1.0
    stream_audio: bool = False
    split_granularity: str = "sentence"
    task_type: str | None = None
    language: str | None = None
    instructions: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    references: list[SpeechReference] | None = None
    x_vector_only_mode: bool | None = None
    stream_codec_output: bool | None = None
    suppress_bootstrap_silence: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    stage_params: dict[str, dict[str, Any]] | None = None


class UploadedVoiceMetadata(BaseModel):
    """Metadata returned for an uploaded TTS voice sample."""

    name: str
    consent: str
    created_at: int
    file_size: int
    mime_type: str
    ref_text: str | None = None
    speaker_description: str | None = None


class VoiceListResponse(BaseModel):
    """Voice registry response for /v1/audio/voices."""

    voices: list[str]
    uploaded_voices: list[UploadedVoiceMetadata]
    cache_stats: dict[str, int] = Field(
        description="API-process uploaded-voice reference cache counters."
    )


class ModelPermission(BaseModel):
    """Model permission info."""

    id: str = "modelperm-default"
    object: str = "model_permission"
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True


class ModelCard(BaseModel):
    """A single model entry."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "sglang-omni"
    permission: list[ModelPermission] = Field(
        default_factory=lambda: [ModelPermission()]
    )
    root: str | None = None


class ModelList(BaseModel):
    """Response for GET /v1/models."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class AdminRequestBase(BaseModel):
    """Common admin request routing controls."""

    stages: list[str] | None = None
    timeout_s: float | None = None


class PauseGenerationRequest(AdminRequestBase):
    mode: str = "abort"


class ContinueGenerationRequest(AdminRequestBase):
    torch_empty_cache: bool = True


class UpdateWeightFromDiskRequest(AdminRequestBase):
    model_path: str
    load_format: str | None = None
    abort_all_requests: bool = False
    weight_version: str | None = None
    is_async: bool = False
    torch_empty_cache: bool = False
    keep_pause: bool = False
    recapture_cuda_graph: bool = False
    token_step: int = 0
    flush_cache: bool = True
    manifest: dict[str, Any] | None = None


class UpdateWeightsFromDistributedRequest(AdminRequestBase):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str = "weight_update_group"
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: str | None = None
    load_format: str | None = None
    torch_empty_cache: bool = False


class InitWeightsUpdateGroupRequest(AdminRequestBase):
    master_address: str
    master_port: int
    world_size: int
    rank_offset: int = 0
    group_name: str = "weight_update_group"
    backend: str = "nccl"


class DestroyWeightsUpdateGroupRequest(AdminRequestBase):
    group_name: str = "weight_update_group"


class WeightsCheckerRequest(AdminRequestBase):
    action: str = "checksum"
