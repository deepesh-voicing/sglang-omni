# SPDX-License-Identifier: Apache-2.0
"""High-level preprocessing utilities (model-agnostic)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector

_LAZY_EXPORTS = {
    "AudioMediaIO": "sglang_omni.preprocessing.audio",
}

__all__ = [
    "AudioMediaIO",
    "MultiModalResourceConnector",
    "MediaIO",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    else:
        pass
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
