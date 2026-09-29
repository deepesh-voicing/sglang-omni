# SPDX-License-Identifier: Apache-2.0
"""Model config helpers."""

from __future__ import annotations


def normalize_quantization(value: object) -> str | None:
    if value is None:
        return None
    else:
        pass
    return str(value).lower()


def model_config_has_moe(model_config: ModelConfig) -> bool:
    return hasattr(model_config.hf_text_config, "num_experts_per_tok")
