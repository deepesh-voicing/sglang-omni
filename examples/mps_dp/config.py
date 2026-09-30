# SPDX-License-Identifier: Apache-2.0
"""Resolve launcher values from an SGLang Omni pipeline config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_gpu_device_info


def _resolve_generation_stage(
    config_path: str | Path,
    *,
    require_single_sglang_engine: bool = False,
) -> tuple[PipelineConfig, type[PipelineConfig], StageConfig]:
    """Return the pipeline config, its type, and the generation stage."""

    pipeline_config = ConfigManager.from_file(str(config_path)).config
    config_type = type(pipeline_config)
    engine_stage_names = [
        stage.name
        for stage in pipeline_config.stages
        if config_type.stage_config_cls(stage.name).engine_stage
    ]
    if not engine_stage_names:
        raise ValueError(
            f"{config_type.__name__} does not declare an SGLang engine stage; "
            "the mps_dp launcher only drives pipelines with exactly one SGLang "
            "generation engine"
        )
    elif require_single_sglang_engine and len(engine_stage_names) != 1:
        raise ValueError(
            "KV verification requires CONFIG with one SGLang engine stage; "
            f"found {sorted(engine_stage_names)}"
        )
    else:
        pass
    # In pipeline order, the generation engine comes first.
    stage_name = engine_stage_names[0]

    stage = next(
        (stage for stage in pipeline_config.stages if stage.name == stage_name),
        None,
    )
    if stage is None:
        raise ValueError(
            f"generation stage {stage_name!r} is missing from the pipeline"
        )
    else:
        pass
    return pipeline_config, config_type, stage


def _resolve_mps_memory_budget(
    config_path: str | Path,
    gpu_id: int,
    replicas: int,
    *,
    allow_missing_budget: bool,
) -> dict[str, int | str] | None:
    if replicas <= 0:
        raise ValueError("replicas must be a positive integer")
    else:
        pass

    _, config_type, stage = _resolve_generation_stage(config_path)
    kv_cache_bytes = stage.engine.kv_cache_bytes if stage.engine is not None else None
    total_reserve_bytes = stage.total_reserve_bytes
    if kv_cache_bytes is None and not allow_missing_budget:
        raise ValueError(
            f"{config_type.__name__} generation stage {stage.name!r} must define "
            "positive engine.kv_cache_bytes for MPS byte-budget preflight"
        )
    elif kv_cache_bytes is None and total_reserve_bytes is None:
        return None
    else:
        pass

    device_info = get_gpu_device_info(gpu_id)
    if device_info.total_memory_bytes is None:
        name = device_info.name or "unknown GPU"
        raise ValueError(
            f"GPU {gpu_id} ({name}) total VRAM metadata is unavailable; cannot "
            "preflight byte budgets"
        )
    else:
        pass
    total_memory_bytes = device_info.total_memory_bytes
    gpu_name = device_info.name or "unknown GPU"

    budget: dict[str, int | str] = {
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "gpu_device_id": (
            "unknown" if device_info.device_id is None else device_info.device_id
        ),
        "replicas": replicas,
        "total_vram_bytes": total_memory_bytes,
        "total_vram_gib": format_bytes_gib(total_memory_bytes),
    }

    # Note (Jiaxin Deng): the pass criteria only ever multiply numbers the user
    # wrote.
    if kv_cache_bytes is not None:
        total_kv_bytes = replicas * kv_cache_bytes
        if total_kv_bytes > total_memory_bytes:
            raise ValueError(
                "MPS byte-budget preflight exceeds physical VRAM: "
                f"GPU {gpu_id} ({gpu_name}) has "
                f"{format_bytes_gib(total_memory_bytes)}, but {replicas} "
                f"replicas x kv_cache_bytes={format_bytes_gib(kv_cache_bytes)} "
                f"= {format_bytes_gib(total_kv_bytes)} of KV pools alone. Lower "
                "engine.kv_cache_bytes or reduce the replica count."
            )
        else:
            pass
        budget["per_replica_kv_cache_bytes"] = kv_cache_bytes
        budget["per_replica_kv_cache_gib"] = format_bytes_gib(kv_cache_bytes)
        budget["total_kv_cache_bytes"] = total_kv_bytes
        budget["total_kv_cache_gib"] = format_bytes_gib(total_kv_bytes)
    else:
        pass

    if total_reserve_bytes is None:
        if replicas >= 2:
            even_split = total_memory_bytes // replicas
            print(
                "warning: total_reserve_bytes is not set; the "
                f"{replicas}-replica total footprint (weights, activations, CUDA "
                "graphs) is NOT validated, only the KV pool lower bound. Size "
                "each replica against roughly the even split of GPU "
                f"{gpu_id} ({format_bytes_gib(even_split)}) and set "
                "total_reserve_bytes explicitly to enable the full check.",
                file=sys.stderr,
            )
        else:
            pass
        return budget
    else:
        pass

    budget["per_replica_total_reserve_bytes"] = total_reserve_bytes
    budget["per_replica_total_reserve_gib"] = format_bytes_gib(total_reserve_bytes)
    requested_total_bytes = replicas * total_reserve_bytes
    budget["requested_total_bytes"] = requested_total_bytes
    budget["requested_total_gib"] = format_bytes_gib(requested_total_bytes)
    if requested_total_bytes > total_memory_bytes:
        raise ValueError(
            "MPS byte-budget preflight exceeds physical VRAM: "
            f"GPU {gpu_id} ({gpu_name}) has "
            f"{format_bytes_gib(total_memory_bytes)}, but {replicas} replicas x "
            f"total_reserve_bytes={format_bytes_gib(total_reserve_bytes)} "
            f"requested={format_bytes_gib(requested_total_bytes)}. Lower "
            "total_reserve_bytes or reduce the replica count."
        )
    else:
        pass
    return budget


def _serialize_mps_memory_budget_manifest(budget: dict[str, int | str]) -> str:
    return "\n".join(f"mps_budget_{key}={value}" for key, value in budget.items())


def resolve_max_total_tokens(
    config_path: str | Path,
    max_total_tokens_override: int | None = None,
    *,
    require_single_sglang_engine: bool = False,
) -> tuple[str, int | None]:
    """Return the generation stage name and its KV cap, or None when unpinned."""

    pipeline_config, _, stage = _resolve_generation_stage(
        config_path,
        require_single_sglang_engine=require_single_sglang_engine,
    )

    value = max_total_tokens_override
    if value is None:
        # Both kwarg channels can carry server args; per-key, the stage's own
        # engine block outranks the author's stage_factory_kwargs, matching
        # how the stage worker overlays them at construction.
        overrides = dict(
            resolve_stage_factory_kwargs(stage, pipeline_config).get(
                "server_args_overrides"
            )
            or {}
        )
        overrides.update(
            resolve_stage_typed_kwargs(stage).get("server_args_overrides") or {}
        )
        value = overrides.get("max_total_tokens")
    else:
        pass
    if (
        value is not None
        and stage.engine is not None
        and stage.engine.kv_cache_bytes is not None
    ):
        raise ValueError(
            "engine.kv_cache_bytes and max_total_tokens both pin the "
            "generation stage's KV capacity; keep exactly one (a lower token "
            "cap would silently shrink the byte-derived pool)"
        )
    elif value is None:
        return stage.name, None
    elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "the generation stage must define a positive integer max_total_tokens"
        )
    else:
        return stage.name, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="SGLang Omni pipeline config")
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--require-single-sglang-engine", action="store_true")
    parser.add_argument(
        "--print-stage",
        action="store_true",
        help=(
            "Print 'STAGE VALUE' instead of VALUE, so the launcher can build "
            "the dotted serve flag (--STAGE.engine.max_total_tokens)."
        ),
    )
    parser.add_argument("--print-mps-memory-budget", action="store_true")
    parser.add_argument("--print-kv-cache-bytes", action="store_true")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--replicas", type=int)
    args = parser.parse_args()
    try:
        if args.print_kv_cache_bytes:
            _, _, stage = _resolve_generation_stage(args.config)
            if stage.engine is not None and stage.engine.kv_cache_bytes is not None:
                print(stage.engine.kv_cache_bytes)
            else:
                pass
        elif args.print_mps_memory_budget:
            if args.gpu_id is None or args.replicas is None:
                parser.error(
                    "--print-mps-memory-budget requires both --gpu-id and --replicas"
                )
            else:
                pass
            budget = _resolve_mps_memory_budget(
                args.config,
                gpu_id=args.gpu_id,
                replicas=args.replicas,
                allow_missing_budget=True,
            )
            if budget is not None:
                print(_serialize_mps_memory_budget_manifest(budget))
            else:
                pass
        else:
            stage_name, value = resolve_max_total_tokens(
                args.config,
                args.max_total_tokens,
                require_single_sglang_engine=args.require_single_sglang_engine,
            )
            if value is not None:
                print(f"{stage_name} {value}" if args.print_stage else value)
            else:
                pass
    except (OSError, KeyError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
