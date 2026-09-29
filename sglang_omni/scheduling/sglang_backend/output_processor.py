# SPDX-License-Identifier: Apache-2.0
"""Converts SGLang GenerationBatchResult to per-request RequestOutputs."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.scheduling.types import RequestOutput, SchedulerOutput


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
        host_token_ids: torch.Tensor | None = None,
    ) -> dict[str, RequestOutput]:
        ids = host_token_ids
        if ids is None:
            ids = model_output.next_token_ids
        else:
            pass
        token_list = ids.tolist() if ids is not None else []

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
            )
        return outputs
