# SPDX-License-Identifier: Apache-2.0
"""Dependency checks must cover the optional evaluation tools selected by CI."""

from importlib import metadata
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3] / ".github/scripts/omni_missing_dependencies.py"
)
SPEC = spec_from_file_location("omni_missing_dependencies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dependencies = module_from_spec(SPEC)
SPEC.loader.exec_module(dependencies)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\ndependencies = ["torch==2.13.0"]\n'
        "[project.optional-dependencies]\n"
        'eval = ["jiwer>=3.0.0", "s3prl>=0.4.18"]\n'
    )
    return path


@pytest.mark.parametrize("s3prl_version", [None, "0.4.17", "0.4.18"])
def test_eval_extra_checks_missing_and_outdated_dependencies(
    project: Path, monkeypatch: pytest.MonkeyPatch, s3prl_version: str | None
) -> None:
    versions = {"torch": "2.13.0", "jiwer": "3.0.0", "s3prl": s3prl_version}

    def version(name: str) -> str:
        installed = versions[name]
        if installed is None:
            raise metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(dependencies.importlib.metadata, "version", version)
    assert dependencies.missing_requirements(project) == []
    assert dependencies.missing_requirements(project, ("eval",)) == (
        [] if s3prl_version == "0.4.18" else ["s3prl>=0.4.18"]
    )


def test_unknown_extra_fails_instead_of_silently_omitting_dependencies(
    project: Path,
) -> None:
    with pytest.raises(KeyError, match="eval-typo"):
        dependencies.missing_requirements(project, ("eval-typo",))
