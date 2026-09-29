# SPDX-License-Identifier: Apache-2.0
"""This file aim to test the model pick logic in the Omni CI workflow.

Author: chenyang zhang https://github.com/zhaochenyang20

In short, if having labels like run-voicing-tts, our CI workflow will
pick that Voicing-TTS preset for TTS. This file aim to test the logic.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OMNI_WORKFLOW = REPO_ROOT / ".github/workflows/omni-ci.yaml"

TTS_LABELS = {
    "voicing-tts": "RUN_VOICING_TTS_LABEL",
    "voicing-tts-custom-voice": "RUN_VOICING_TTS_CUSTOM_VOICE_LABEL",
}


def tts_pick_script() -> str:
    jobs = yaml.load(OMNI_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)[
        "jobs"
    ]
    return next(
        step["run"]
        for step in jobs["pick-tts-model"]["steps"]
        if step.get("id") == "tts"
    )


def run_tts_pick(
    tmp_path: Path, labels: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    github_output = tmp_path / "github_output"
    github_output.touch()
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(github_output),
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "sgl-project/sglang-omni",
        "EXACT_SHA": "deadbeef",
        "TTS_CI_MODEL_OVERRIDE": "",
        "RUN_VOICING_TTS_LABEL": "false",
        "RUN_VOICING_TTS_CUSTOM_VOICE_LABEL": "false",
        **labels,
    }
    return subprocess.run(
        ["bash", "-c", tts_pick_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_tts_pick_rotates_without_labels(tmp_path: Path) -> None:
    result = run_tts_pick(tmp_path, {})
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Selection digest for TTS CI model:" in result.stdout
    assert "Selected TTS CI model: voicing-tts" in result.stdout


@pytest.mark.parametrize("tts_model,tts_label", list(TTS_LABELS.items()))
def test_tts_pick_honors_labels(tmp_path: Path, tts_model: str, tts_label: str) -> None:
    result = run_tts_pick(tmp_path, {tts_label: "true"})
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Selected TTS CI model: {tts_model}" in result.stdout
    assert "Selection digest for TTS CI model:" not in result.stdout
    assert f"selected_model={tts_model}\n" in (tmp_path / "github_output").read_text()


def test_tts_pick_rejects_conflicting_labels(tmp_path: Path) -> None:
    result = run_tts_pick(
        tmp_path,
        {
            "RUN_VOICING_TTS_LABEL": "true",
            "RUN_VOICING_TTS_CUSTOM_VOICE_LABEL": "true",
        },
    )
    assert result.returncode != 0
    assert (tmp_path / "github_output").read_text() == ""


@pytest.fixture
def slash_handler(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(
        sys.modules, "github", SimpleNamespace(Auth=Mock(), Github=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "github.GithubException",
        SimpleNamespace(GithubException=Exception),
    )
    spec = spec_from_file_location(
        "slash_command_handler", REPO_ROOT / "scripts/ci/utils/slash_command_handler.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slash_targets_select_a_tts_preset(slash_handler: ModuleType) -> None:
    assert slash_handler.parse_model_targets(
        "/tag-and-rerun-ci voicing-tts-custom-voice".split()
    ) == ("voicing-tts-custom-voice", None)


def test_slash_targets_reject_conflicting_models(slash_handler: ModuleType) -> None:
    model, error = slash_handler.parse_model_targets(
        "/tag-run-ci-label voicing-tts voicing-tts-custom-voice".split()
    )
    assert model is None
    assert error is not None


@pytest.mark.parametrize(
    "selected,previous",
    [
        ("voicing-tts", "voicing-tts-custom-voice"),
        ("voicing-tts-custom-voice", "voicing-tts"),
    ],
)
def test_slash_tag_replaces_the_other_tts_label(
    slash_handler: ModuleType, selected: str, previous: str
) -> None:
    pr, comment = Mock(), Mock()
    pr.get_labels.return_value = [SimpleNamespace(name=f"run-{previous}")]
    assert slash_handler.handle_tag_run_ci(
        pr, comment, {"can_tag_run_ci_label": True}, tts_model_target=selected
    )
    pr.remove_from_labels.assert_called_once_with(f"run-{previous}")
    assert pr.add_to_labels.call_args_list == [call("run-ci"), call(f"run-{selected}")]
    comment.create_reaction.assert_called_once_with("+1")


@pytest.mark.parametrize("command", ["/tag-run-ci-label", "/tag-and-rerun-ci"])
def test_slash_commands_forward_tts_target(
    slash_handler: ModuleType, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    environment = {
        "GITHUB_TOKEN": "test-token",
        "REPO_FULL_NAME": "owner/repo",
        "PR_NUMBER": "1",
        "COMMENT_ID": "2",
        "COMMENT_BODY": f"{command} voicing-tts-custom-voice",
        "USER_LOGIN": "contributor",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        slash_handler,
        "load_permissions",
        Mock(return_value={"can_tag_run_ci_label": True, "can_rerun_failed_ci": True}),
    )
    tagged = Mock(return_value=True)
    rerun = Mock(return_value=True)
    monkeypatch.setattr(slash_handler, "handle_tag_run_ci", tagged)
    monkeypatch.setattr(slash_handler, "handle_rerun_failed_ci", rerun)
    monkeypatch.setattr(slash_handler.time, "sleep", Mock())
    slash_handler.main()
    assert tagged.call_args.kwargs["tts_model_target"] == "voicing-tts-custom-voice"
    if command == "/tag-and-rerun-ci":
        assert rerun.call_args.kwargs["force_full_omni_ci_rerun"] is True
    else:
        rerun.assert_not_called()
