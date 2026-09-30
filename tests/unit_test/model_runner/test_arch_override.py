# SPDX-License-Identifier: Apache-2.0
"""The KV pool of the talker engine is sized from the talker's layers.

SGLang builds the engine's ModelConfig from the root checkpoint config, so
the Voicing-TTS talker engine can start from a root text config whose layer
counts differ from the talker's. SGLang then sizes the pool from the larger
of num_hidden_layers and num_attention_layers through resolve_layer_indices,
so the override has to leave the pool at the talker's layers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from sglang.srt.model_executor.model_runner_components.layer_setup import (  # noqa: E402
    resolve_layer_indices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

from sglang_omni.model_runner.model_worker import ModelWorker  # noqa: E402


def voicing_tts_engine_config():
    root_text = SimpleNamespace(
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=128,
        vocab_size=151936,
    )
    talker_config = SimpleNamespace(
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_size=2048,
        head_dim=128,
        vocab_size=3072,
    )
    hf_config = SimpleNamespace(
        architectures=["VoicingTTSForConditionalGeneration"],
        talker_config=talker_config,
    )
    return SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=root_text,
        num_attention_heads=root_text.num_attention_heads,
        num_key_value_heads=root_text.num_key_value_heads,
        hidden_size=root_text.hidden_size,
        num_hidden_layers=root_text.num_hidden_layers,
        num_attention_layers=root_text.num_hidden_layers,
        num_nextn_predict_layers=None,
        head_dim=root_text.head_dim,
        vocab_size=root_text.vocab_size,
    )


def pool_layers(config) -> int:
    return resolve_layer_indices(
        model=object(),
        model_config=config,
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    ).num_effective_layers


def test_talker_pool_is_sized_from_the_talker_layers() -> None:
    config = voicing_tts_engine_config()
    assert pool_layers(config) == 36
    ModelWorker.apply_arch_override(config, "VoicingTTSTalker")
    assert pool_layers(config) == 28
    assert config.hidden_size == 2048
    assert config.hf_config.architectures == ["VoicingTTSTalker"]


def test_unknown_arch_override_only_renames_the_architecture() -> None:
    config = voicing_tts_engine_config()
    ModelWorker.apply_arch_override(config, "UnknownTalker")
    assert pool_layers(config) == 36
    assert config.hf_config.architectures == ["UnknownTalker"]
