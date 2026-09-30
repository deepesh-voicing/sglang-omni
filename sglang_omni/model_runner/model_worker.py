from __future__ import annotations

import logging
import os
import socket
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from sglang_omni.platforms import current_platform
from sglang_omni.utils.misc import model_config_has_moe
from sglang_omni.vendor.sglang.server_args import override_server_args

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs
else:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ModelWorkerConfig:
    model_arch_override: str | None = None
    weight_prefix: str | None = None
    nccl_port: int | None = None
    total_gpu_memory_fraction: float | None = None
    kv_cache_bytes: int | None = None
    enable_prefill_input_embeds: bool = False


@dataclass(slots=True)
class PrefillCudaGraphUsage:
    replay_count: int = 0
    standard_eager_count: int = 0
    custom_eager_count: int = 0
    replay_buckets: Counter[int] = field(default_factory=Counter)


_ARCH_CONFIG_MAP: dict[str, tuple[str, str | None]] = {
    "VoicingTTSTalker": ("talker_config", None),
}


class ModelWorker:
    def __init__(
        self,
        config: ModelWorkerConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int = 0,
    ):
        self.server_args = server_args
        self.model_arch_override = config.model_arch_override
        self.weight_prefix = config.weight_prefix
        self.nccl_port = config.nccl_port
        self.total_gpu_memory_fraction = config.total_gpu_memory_fraction
        self.kv_cache_bytes = config.kv_cache_bytes
        self.enable_prefill_input_embeds = config.enable_prefill_input_embeds

        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.init_model_config()
        effective_quantization = self.configure_backend_policy()
        from sglang.srt.runtime_context import publish

        publish(self.server_args, role="scheduler")
        initialize_model_worker_backend_globals(
            self.model_config, effective_quantization
        )
        self.init_model_runner()
        self.prefill_cuda_graph_usage = PrefillCudaGraphUsage()

        self.device = self.model_runner.device
        from sglang.srt.runtime_context import get_device
        from sglang.srt.utils import broadcast_pyobj, set_random_seed

        self.random_seed = broadcast_pyobj(
            [get_device().random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

    def init_model_config(self):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            server_args=self.server_args,
            is_draft_model=False,
        )

        if self.model_arch_override is not None:
            self.apply_arch_override(self.model_config, self.model_arch_override)
        else:
            pass

    @staticmethod
    def apply_arch_override(model_config: ModelConfig, arch: str) -> None:
        """Override model config for a sub-model architecture."""
        model_config.hf_config.architectures = [arch]
        entry = _ARCH_CONFIG_MAP.get(arch)
        if entry is None:
            return
        else:
            pass
        sub_config_attr, text_config_attr = entry
        sub_cfg = getattr(model_config.hf_config, sub_config_attr, None)
        if sub_cfg is None:
            return
        else:
            pass
        text_cfg = getattr(sub_cfg, text_config_attr) if text_config_attr else sub_cfg
        model_config.hf_text_config = text_cfg
        model_config.num_attention_heads = text_cfg.num_attention_heads
        model_config.num_key_value_heads = text_cfg.num_key_value_heads
        model_config.hidden_size = text_cfg.hidden_size
        model_config.num_hidden_layers = text_cfg.num_hidden_layers
        # note(ratish): SGLang sizes the KV pool from the larger of these two
        # and set the second from the root text config at construction.
        model_config.num_attention_layers = text_cfg.num_hidden_layers

    def configure_backend_policy(self) -> str | None:
        return current_platform.apply_model_worker_backend_policy(
            self.server_args,
            self.model_config,
            self.model_arch_override,
        )

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def get_worker_info(self):
        max_total_num_tokens = self.model_runner.max_total_num_tokens
        effective_max_total_num_tokens = (
            self.model_runner.effective_max_total_num_tokens
        )
        max_req_len = min(
            self.server_args.context_length - 1,
            effective_max_total_num_tokens - 1,
        )
        max_req_input_len = max_req_len - 1
        req_pool = self.model_runner.req_to_token_pool
        kv_pool = self.model_runner.token_to_kv_pool_allocator
        max_running_requests = self.model_runner.max_running_requests
        return (
            max_total_num_tokens,
            self.server_args.max_prefill_tokens,
            max_running_requests,
            self.server_args.max_queued_requests,
            max_req_len,
            max_req_input_len,
            self.random_seed,
            self.device,
            req_pool.size,
            req_pool.max_context_len,
            kv_pool.size,
        )

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_attention_tp_cpu_group(self):
        return self.model_runner.attention_tp_group.cpu_group

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def init_model_runner(self):
        from .sglang_model_runner import SGLModelRunner

        nccl_port = (
            self.nccl_port if self.nccl_port is not None else resolve_nccl_port()
        )
        self.model_runner = SGLModelRunner(
            model_config=self.model_config,
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=nccl_port,
            model_arch_override=self.model_arch_override,
            weight_prefix=self.weight_prefix,
            total_gpu_memory_fraction=self.total_gpu_memory_fraction,
            kv_cache_bytes=self.kv_cache_bytes,
        )

    def forward_batch_generation(self, forward_batch):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        out = self.model_runner.forward(forward_batch=forward_batch)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        self.record_prefill_cuda_graph_usage(
            forward_batch,
            can_run_graph=bool(can_run_cuda_graph),
        )
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        return batch_result

    def record_prefill_cuda_graph_usage(
        self,
        forward_batch: Any,
        *,
        can_run_graph: bool,
    ) -> None:
        mode = forward_batch.forward_mode
        if not mode.is_extend() or mode.is_cuda_graph():
            return
        else:
            pass

        if not can_run_graph:
            # Note (wenyao): custom eager forwards (visual/deepstack) return
            # before ModelWorker is called; intentionally absent here.
            self.prefill_cuda_graph_usage.standard_eager_count += 1
            return
        else:
            pass

        runner = self.model_runner.prefill_cuda_graph_runner
        buckets = runner.capture_num_tokens
        actual_bucket = buckets[bisect_left(buckets, len(forward_batch.input_ids))]
        self.prefill_cuda_graph_usage.replay_count += 1
        self.prefill_cuda_graph_usage.replay_buckets[int(actual_bucket)] += 1

    def prefill_cuda_graph_info(self) -> dict[str, Any]:
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        runner = self.model_runner.prefill_cuda_graph_runner
        if isinstance(runner, PrefillCudaGraphRunner):
            capture_num_tokens = [int(value) for value in runner.capture_num_tokens]
            backend_runner = type(runner.backend).__name__
            input_embeds_slot = runner.buffer_registry.has_slot("input_embeds")
        else:
            capture_num_tokens, backend_runner, input_embeds_slot = None, None, False
        from sglang.srt.runtime_context import get_exec

        backend = get_exec().graph.cuda_graph_config.prefill.backend
        usage = self.prefill_cuda_graph_usage
        return {
            "backend": backend,
            "runner": type(runner).__name__ if runner is not None else None,
            "backend_runner": backend_runner,
            "capture_num_tokens": capture_num_tokens,
            "input_embeds_slot": input_embeds_slot,
            "replay_count": int(usage.replay_count),
            "standard_eager_count": int(usage.standard_eager_count),
            "custom_eager_count": int(usage.custom_eager_count),
            "replay_buckets": {
                str(bucket): int(count)
                for bucket, count in sorted(usage.replay_buckets.items())
            },
        }

    def model_info(self) -> dict[str, Any]:
        from sglang.srt.runtime_context import get_model, get_parallel, get_serving

        return {
            "model_path": get_model().model_path,
            "load_format": get_model().load_format,
            "weight_version": get_serving().weight_version,
            "tp_rank": self.tp_rank,
            "tp_size": get_parallel().tp_size,
            "model_arch_override": self.model_arch_override,
            "supports_weight_update": True,
            "supports_weight_checker": True,
            "prefill_cuda_graph": self.prefill_cuda_graph_info(),
        }

    def update_weights_from_disk(self, payload: dict[str, Any]) -> tuple[bool, str]:
        model_path = payload.get("model_path")
        if not model_path:
            return False, "model_path is required"
        else:
            pass
        from sglang.srt.runtime_context import get_model

        update = self.model_runner.update_weights_from_disk
        load_format = payload.get("load_format") or get_model().load_format
        success, message = update(
            model_path,
            load_format,
            recapture_cuda_graph=bool(payload.get("recapture_cuda_graph", False)),
        )
        # The runner's WeightUpdater already records model_path and
        # load_format in the model bag; weight_version is omni's own field.
        weight_version = payload.get("weight_version")
        if success and weight_version is not None:
            override_server_args(
                self.server_args,
                "sglang-omni-weight-update-disk",
                weight_version=weight_version,
            )
        else:
            pass
        return bool(success), str(message)

    def update_weights_from_tensor(self, payload: dict[str, Any]) -> tuple[bool, str]:
        if payload.get("serialized_named_tensors") is not None:
            return (
                False,
                "update_weights_from_tensor requires a tensor data plane; "
                "Omni admin control plane only carries metadata",
            )
        else:
            pass
        return self.call_optional_weight_method("update_weights_from_tensor", payload)

    def init_weights_update_group(self, payload: dict[str, Any]) -> tuple[bool, str]:
        init = self.model_runner.init_weights_update_group
        master_address = payload.get("master_address")
        master_port = payload.get("master_port")
        world_size = payload.get("world_size")
        if not master_address or master_port is None or world_size is None:
            return False, "master_address, master_port and world_size are required"
        else:
            pass
        try:
            master_port_int = int(master_port)
            rank_offset_int = int(payload.get("rank_offset", 0))
            world_size_int = int(world_size)
        except (TypeError, ValueError):
            return False, "master_port, rank_offset and world_size must be integers"
        success, message = init(
            master_address,
            master_port_int,
            rank_offset_int,
            world_size_int,
            payload.get("group_name") or "weight_update_group",
            backend=payload.get("backend") or "nccl",
        )
        return bool(success), str(message)

    def destroy_weights_update_group(self, payload: dict[str, Any]) -> tuple[bool, str]:
        destroy = self.model_runner.destroy_weights_update_group
        success, message = destroy(payload.get("group_name") or "weight_update_group")
        return bool(success), str(message)

    def update_weights_from_distributed(
        self, payload: dict[str, Any]
    ) -> tuple[bool, str]:
        update = self.model_runner.update_weights_from_distributed
        names = payload.get("names")
        dtypes = payload.get("dtypes")
        shapes = payload.get("shapes")
        if names is None or dtypes is None or shapes is None:
            return False, "names, dtypes and shapes are required"
        else:
            pass
        # Pydantic already guards type/None at the HTTP boundary; this length
        # check is the one guard that matters — sglang zips names/dtypes/shapes
        # and silently truncates to the shortest, under-broadcasting weights.
        name_count = len(names)
        dtype_count = len(dtypes)
        shape_count = len(shapes)
        if name_count == 0 or dtype_count == 0 or shape_count == 0:
            return False, "names, dtypes and shapes must be non-empty"
        else:
            pass
        if name_count != dtype_count or name_count != shape_count:
            return False, "names, dtypes and shapes must have the same length"
        else:
            pass
        success, message = update(
            names,
            dtypes,
            shapes,
            payload.get("group_name") or "weight_update_group",
            load_format=payload.get("load_format"),
        )
        if success:
            weight_version = payload.get("weight_version")
            if weight_version is not None:
                override_server_args(
                    self.server_args,
                    "sglang-omni-weight-update-distributed",
                    weight_version=weight_version,
                )
            else:
                pass
        else:
            pass
        return bool(success), str(message)

    def weights_checker(self, action: str) -> dict[str, Any]:
        checker = getattr(self, "strict_weight_checker", None)
        if checker is None:
            from sglang_omni.model_runner.weight_checker import StrictWeightChecker

            checker = StrictWeightChecker(self.model_runner)
            self.strict_weight_checker = checker
        else:
            pass
        return checker.run(action)

    def call_optional_weight_method(
        self,
        method_name: str,
        payload: dict[str, Any],
    ) -> tuple[bool, str]:
        method = getattr(self.model_runner, method_name)
        recv_req = SimpleNamespace(**payload)
        success, message = method(recv_req)
        return bool(success), str(message)


def resolve_nccl_port() -> int:
    master_port = os.environ.get("MASTER_PORT")
    if master_port:
        return int(master_port)
    else:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = sock.getsockname()[1]
    except PermissionError:
        # Some restricted CI / sandbox environments do not allow ephemeral socket
        # binding during test-time configuration. Fall back to a stable default so
        # callers still receive a valid NCCL port choice.
        port = 29500

    os.environ["MASTER_PORT"] = str(port)
    return port


def initialize_model_worker_backend_globals(
    model_config: ModelConfig,
    effective_quantization: str | None,
) -> None:
    """Initialize backend globals needed by direct workers before model loading.

    Both initializers read the published config bags, so this runs after publish.
    """

    if model_config_has_moe(model_config):
        from sglang.srt.layers.moe import initialize_moe_config

        initialize_moe_config()
    else:
        pass

    if effective_quantization == "fp8":
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config

        initialize_fp8_gemm_config()
    else:
        pass
