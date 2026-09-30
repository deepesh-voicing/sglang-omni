# SPDX-License-Identifier: Apache-2.0
"""Client.speech() result fields and the GenerateRequest input contract."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sglang_omni.client import Client
from sglang_omni.client.client import extract_inputs
from sglang_omni.client.types import GenerateRequest


class SubmitStubCoordinator:
    """Non-streaming coordinator stub: speech() only needs submit()."""

    def __init__(self, result: Any) -> None:
        self.result = result

    async def submit(self, request_id: str, omni_request: Any) -> Any:
        del request_id, omni_request
        return self.result


def test_speech_surfaces_finish_reason() -> None:
    client = Client(
        SubmitStubCoordinator(
            {
                "audio_data": [0.0, 0.1, -0.1],
                "sample_rate": 24000,
                "finish_reason": "length",
            }
        )
    )

    result = asyncio.run(
        client.speech(
            GenerateRequest(prompt="hello"),
            request_id="speech-1",
            response_format="pcm",
        )
    )

    assert result.finish_reason == "length"
    assert result.mime_type == "audio/pcm"


def test_extract_inputs_requires_prompt() -> None:
    assert extract_inputs(GenerateRequest(prompt="hi")) == "hi"
    with pytest.raises(ValueError, match="requires a prompt"):
        extract_inputs(GenerateRequest())
