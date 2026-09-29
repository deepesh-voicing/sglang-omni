# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest import mock

import sglang.srt.models.registry as registry_mod

import sglang_omni.model_runner.sglang_model_runner as runner_mod


def test_register_omni_model_registers_the_voicing_tts_talker(monkeypatch):
    registry = SimpleNamespace(models={})
    monkeypatch.setattr(registry_mod, "ModelRegistry", registry)
    monkeypatch.setattr(importlib, "import_module", lambda name: mock.MagicMock())

    runner_mod.SGLModelRunner.register_omni_model(object())

    assert set(registry.models) == {"VoicingTTSTalker"}


def test_register_omni_model_skips_unimportable(monkeypatch):
    registry = SimpleNamespace(models={})
    monkeypatch.setattr(registry_mod, "ModelRegistry", registry)

    def fake_import(name, *args, **kwargs):
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    runner_mod.SGLModelRunner.register_omni_model(object())

    assert registry.models == {}
