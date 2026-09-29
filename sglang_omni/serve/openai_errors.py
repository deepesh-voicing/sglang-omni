# SPDX-License-Identifier: Apache-2.0
"""Shared OpenAI-compatible API error classification helpers."""

from __future__ import annotations

import re

_BAD_REQUEST_MARKERS = (
    "longer than the model's context length",
    "Requested token count exceeds the model's maximum context length",
    "Request requires more tokens than the KV cache can hold",
)
_BAD_REQUEST_PATTERNS = (
    re.compile(r"^Request\s+\S+\s+exceeds the maximum number of tokens:"),
    re.compile(r"^Request\s+\S+\s+requires too many SWA KV tokens for"),
)


def is_bad_request_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS) or any(
        pattern.search(message) is not None for pattern in _BAD_REQUEST_PATTERNS
    )
