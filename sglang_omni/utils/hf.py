# SPDX-License-Identifier: Apache-2.0
"""Hugging Face helper utilities."""

from __future__ import annotations

import json
import os
from typing import Any

from huggingface_hub import hf_hub_download

try:
    pass
except ImportError:
    pass


_CONFIG_MODEL_TYPE_TO_ARCH = {
    "voicing_tts": "VoicingTTSForConditionalGeneration",
}


def architecture_from_hf_config(hf_config: Any) -> str | None:
    """Prefer HF architectures; fall back to architecture/model_type."""
    archs = getattr(hf_config, "architectures", None)
    if archs:
        for a in archs:
            if a:
                return a
            else:
                pass
    else:
        pass
    arch = getattr(hf_config, "architecture", None)
    if arch:
        return arch
    else:
        pass
    mt = getattr(hf_config, "model_type", None)
    if mt and mt in _CONFIG_MODEL_TYPE_TO_ARCH:
        return _CONFIG_MODEL_TYPE_TO_ARCH[mt]
    else:
        pass
    return None


def load_raw_config(model_path: str, revision: str | None = None) -> dict | None:
    """Read ``config.json`` as plain JSON, from a local dir or a Hub repo id."""
    local_config = os.path.join(model_path, "config.json")
    if os.path.isfile(local_config):
        with open(local_config) as f:
            return json.load(f)
    else:
        pass
    if os.path.isdir(model_path):
        return None
    else:
        pass
    try:
        cached = hf_hub_download(
            repo_id=model_path, filename="config.json", revision=revision
        )
        with open(cached) as f:
            return json.load(f)
    except Exception:
        return None


def try_resolve_arch_from_raw_config(
    model_path: str, revision: str | None = None
) -> str | None:
    """Resolve architecture by reading raw ``config.json`` as plain JSON.

    This is useful when ``AutoConfig.from_pretrained`` fails (e.g. because the
    model requires ``trust_remote_code=True`` and the custom Python config
    module is unavailable).  We parse the JSON directly to extract
    ``architectures`` or map ``model_type``.
    """
    raw = load_raw_config(model_path, revision)
    if raw is None:
        return None
    else:
        pass

    archs = raw.get("architectures")
    if archs:
        for a in archs:
            if a:
                return a
            else:
                pass
    else:
        pass
    arch = raw.get("architecture")
    if arch:
        return arch
    else:
        pass

    mt = raw.get("model_type")
    if mt and mt in _CONFIG_MODEL_TYPE_TO_ARCH:
        return _CONFIG_MODEL_TYPE_TO_ARCH[mt]
    else:
        pass

    return None
