from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "find_available_port": ("sglang_omni.utils.connection", "find_available_port"),
    "architecture_from_hf_config": (
        "sglang_omni.utils.hf",
        "architecture_from_hf_config",
    ),
    "try_resolve_arch_from_raw_config": (
        "sglang_omni.utils.hf",
        "try_resolve_arch_from_raw_config",
    ),
    "import_string": ("sglang_omni.utils.imports", "import_string"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
