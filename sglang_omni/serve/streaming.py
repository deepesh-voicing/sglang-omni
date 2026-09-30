# SPDX-License-Identifier: Apache-2.0
"""Shared streaming response helpers for OpenAI-compatible endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


async def close_async_iterator_if_supported(stream: AsyncIterator[Any]) -> None:
    try:
        close = stream.aclose
    except AttributeError:
        return
    await close()
