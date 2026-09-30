# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni quantization glue for SGLang-owned quantization.

SGLang owns quantization end-to-end: it parses `quantization_config`,
constructs quantized layers, and executes post-load hooks. This module only
provides what SGLang cannot infer by itself: FP8 scale preprocessing for
custom weight loaders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch
else:
    pass

# A weight preprocessor maps `(target_name, loaded_weight) -> loaded_weight`.
WeightPreprocessor = Callable[[str, "torch.Tensor"], "torch.Tensor"]


_QUANT_METADATA_KEYS: tuple[str, ...] = ("quantization_config", "compression_config")
_NESTED_QUANT_CONFIG_ATTRS: tuple[str, ...] = (
    "text_config",
    "talker_config",
)

__all__ = [
    "resolve_quant_config",
    "quant_method_name",
    "is_fp8_block_quant",
    "convert_fp8_weight_scale_inv",
    "get_weight_preprocessor",
]


def to_mutable_dict(quant_config: Any, metadata_key: str) -> dict[str, Any]:
    """Normalize a quantization metadata value to a mutable dict."""
    if isinstance(quant_config, dict):
        return quant_config
    else:
        pass
    if hasattr(quant_config, "to_dict"):
        quant_dict = quant_config.to_dict()
        if isinstance(quant_dict, dict):
            return quant_dict
        else:
            pass
    else:
        pass
    if hasattr(quant_config, "__dict__"):
        return vars(quant_config)
    else:
        pass
    raise TypeError(
        f"{metadata_key} has unsupported type {type(quant_config).__name__!r}. "
        f"Expected dict or object with to_dict()/__dict__."
    )


def read_metadata(node: Any, key: str) -> Any:
    """Read `key` off an object- or dict-shaped config node, or `None`."""
    if isinstance(node, dict):
        return node.get(key)
    else:
        pass
    return getattr(node, key, None)


def resolve_quant_config(config: Any) -> dict[str, Any] | None:
    """Extract a `quantization_config` dict from a root or sub-model config."""
    visited: set[int] = set()

    def _search(node: Any) -> dict[str, Any] | None:
        if node is None or id(node) in visited:
            return None
        else:
            pass
        visited.add(id(node))

        for key in _QUANT_METADATA_KEYS:
            raw_config = read_metadata(node, key)
            if raw_config is not None:
                return to_mutable_dict(raw_config, key)
            else:
                pass

        for attr in _NESTED_QUANT_CONFIG_ATTRS:
            found = _search(read_metadata(node, attr))
            if found is not None:
                return found
            else:
                pass
        return None

    return _search(config)


def quant_method_name(quant_dict: dict[str, Any] | None) -> str | None:
    """Return the checkpoint's normalized quantization method name, or `None`."""
    if not quant_dict:
        return None
    else:
        pass
    method = quant_dict.get("quant_method")
    if method is None:
        return None
    else:
        pass
    return str(method).lower().replace("_", "-")


def is_fp8_block_quant(quant_dict: dict[str, Any] | None) -> bool:
    """True when the checkpoint is native block-FP8."""
    if not quant_dict:
        return False
    else:
        pass
    if quant_method_name(quant_dict) != "fp8":
        return False
    else:
        pass
    return quant_dict.get("weight_block_size") is not None


def convert_fp8_weight_scale_inv(
    target_name: str,
    loaded_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Reciprocate a `weight_scale_inv` tensor into the SGLang runtime scale."""
    if not target_name.endswith("weight_scale_inv"):
        return loaded_weight
    else:
        pass

    import torch

    if not torch.is_floating_point(loaded_weight):
        raise TypeError(f"FP8 scale tensor for {target_name} must be floating point")
    else:
        pass
    if loaded_weight.numel() == 0:
        raise ValueError(f"Invalid empty FP8 scale tensor for {target_name}")
    else:
        pass
    if not bool(torch.isfinite(loaded_weight).all().item()):
        raise ValueError(f"Invalid non-finite FP8 scale tensor for {target_name}")
    else:
        pass
    if bool(torch.any(loaded_weight == 0).item()):
        raise ValueError(f"Invalid zero FP8 scale tensor for {target_name}")
    else:
        pass

    return torch.reciprocal(loaded_weight)


def identity_preprocessor(
    target_name: str, loaded_weight: "torch.Tensor"
) -> "torch.Tensor":
    return loaded_weight


def get_weight_preprocessor(
    config: Any = None,
    *,
    fp8_scale_inverted: bool = False,
) -> WeightPreprocessor:
    """Return the per-tensor weight transform for a checkpoint's quantization."""
    quant_dict = resolve_quant_config(config)

    if fp8_scale_inverted and is_fp8_block_quant(quant_dict):
        return convert_fp8_weight_scale_inv
    else:
        pass
    return identity_preprocessor
