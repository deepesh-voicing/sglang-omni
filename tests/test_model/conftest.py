# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and hooks for test_model tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

pytest_plugins = ["tests.utils"]

if TYPE_CHECKING:
    from typing import Generator


def parse_cpuset(spec: str) -> set[int]:
    """Parse a Linux cpulist such as "0-23,64-87" into a CPU id set."""
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Empty component in OMNI_CI_CPUSET {spec!r}")
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if lo > hi:
                raise ValueError(f"Invalid OMNI_CI_CPUSET range {part!r}")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f"OMNI_CI_CPUSET {spec!r} selects no CPUs")
    return cpus


def apply_omni_ci_cpuset() -> set[int] | None:
    spec = os.environ.get("OMNI_CI_CPUSET", "").strip()
    if not spec or not hasattr(os, "sched_setaffinity"):
        return None
    requested = parse_cpuset(spec)
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, requested)
    # Note: (Jiaxin Deng) Linux may silently intersect the request with the
    # cgroup/cpuset-allowed CPUs; a partial pin would invalidate calibration.
    effective = os.sched_getaffinity(0)
    if effective != requested:
        raise RuntimeError(
            f"OMNI_CI_CPUSET pinning ineffective: requested {sorted(requested)}, "
            f"previously allowed {sorted(previous)}, effective {sorted(effective)}"
        )
    return requested


@pytest.fixture(autouse=True, scope="session")
def pin_omni_ci_cpuset() -> "Generator[None, None, None]":
    """Pin the test session, and every server it spawns, to OMNI_CI_CPUSET.

    The gates in this directory measure host-bound serving stacks, so on a
    shared runner concurrent jobs inflate per-request CPU cost and shift
    calibrated floors. Child processes inherit the affinity, which keeps the
    managed router, its workers, and the bench client on the reserved cores.
    Pinning restrains only this session, so a contention sampler reports
    foreign load on the reserved cores at session end. The report is
    advisory, never fatal: contention can only depress perf numbers, so a
    gate that passed under intrusion passed for real, and a gate that failed
    carries the contention line for triage while retrying through the normal
    failure path. Calibration is stricter and rejects the round itself.
    """
    cpus = apply_omni_ci_cpuset()
    if cpus is None:
        yield
        return
    from tests.utils.ci_cpu_contention import ContentionSampler

    sampler = ContentionSampler(cpus)
    sampler.start()
    try:
        yield
    finally:
        sampler.stop()
        print(sampler.summary())


TTS_ALLOWED_CONCURRENCIES = (1, 2, 4, 8, 16)
TTS_STAGE_NONSTREAM = "tts-stage-1-nonstream"
TTS_STAGE_STREAM = "tts-stage-2-stream"
TTS_STAGE_CONSISTENCY = "tts-stage-3-consistency"
TTS_CI_STAGES = (
    TTS_STAGE_NONSTREAM,
    TTS_STAGE_STREAM,
    TTS_STAGE_CONSISTENCY,
)
TTS_FULL_SWEEP_VALUE = "all"
TTS_STAGE_ALL = "all"
TTS_CONCURRENCY_OPTION = "--concurrency"
SELECTED_TTS_CONCURRENCIES = pytest.StashKey[tuple[int, ...]]()
TTS_STAGE_OPTION = "--tts-stage"
SELECTED_TTS_CI_STAGE = pytest.StashKey[str]()
TTS_CI_MODEL_OPTION = "--tts-ci-model"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        TTS_CONCURRENCY_OPTION,
        action="store",
        default="16",
        help=(
            "Select the TTS benchmark concurrency. "
            "Use one of {1,2,4,8,16} or 'all' for the full sweep."
        ),
    )
    parser.addoption(
        TTS_STAGE_OPTION,
        action="store",
        default=TTS_STAGE_ALL,
        help=(
            f"Select the TTS CI stage. Use one of {TTS_CI_STAGES} or '{TTS_STAGE_ALL}'."
        ),
    )
    parser.addoption(
        TTS_CI_MODEL_OPTION,
        action="store",
        default="",
        help=(
            "Select the TTS CI model preset. "
            "Use one of the presets in tests/test_model/tts_ci_config.py. "
            "If omitted, use TTS_CI_MODEL from the environment."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    option_value = config.getoption(TTS_CONCURRENCY_OPTION)
    config.stash[SELECTED_TTS_CONCURRENCIES] = parse_tts_concurrency(option_value)
    stage_value = config.getoption(TTS_STAGE_OPTION)
    config.stash[SELECTED_TTS_CI_STAGE] = parse_tts_ci_stage(stage_value)
    tts_model_value = config.getoption(TTS_CI_MODEL_OPTION)
    if tts_model_value:
        os.environ["TTS_CI_MODEL"] = parse_tts_ci_model(tts_model_value)


@pytest.fixture(scope="session")
def selected_tts_concurrencies(
    pytestconfig: pytest.Config,
) -> tuple[int, ...]:
    return pytestconfig.stash[SELECTED_TTS_CONCURRENCIES]


@pytest.fixture(scope="session")
def selected_tts_ci_stage(pytestconfig: pytest.Config) -> str:
    return pytestconfig.stash[SELECTED_TTS_CI_STAGE]


def parse_tts_concurrency(option_value: str) -> tuple[int, ...]:
    normalized_value = option_value.strip().lower()
    if normalized_value == TTS_FULL_SWEEP_VALUE:
        return TTS_ALLOWED_CONCURRENCIES

    try:
        concurrency = int(normalized_value)
    except ValueError as exc:
        raise pytest.UsageError(
            "Invalid value for --concurrency. Use one of {1,2,4,8,16} or 'all'."
        ) from exc

    if concurrency not in TTS_ALLOWED_CONCURRENCIES:
        raise pytest.UsageError(
            f"Unsupported concurrency {concurrency}. "
            f"Use one of {TTS_ALLOWED_CONCURRENCIES} or 'all'."
        )
    return (concurrency,)


def parse_tts_ci_stage(option_value: str) -> str:
    normalized_value = option_value.strip().lower()
    if normalized_value == TTS_STAGE_ALL:
        return TTS_STAGE_ALL
    if normalized_value not in TTS_CI_STAGES:
        raise pytest.UsageError(
            f"Unsupported value for {TTS_STAGE_OPTION}: {option_value!r}. "
            f"Use one of {TTS_CI_STAGES} or '{TTS_STAGE_ALL}'."
        )
    return normalized_value


def parse_tts_ci_model(option_value: str) -> str:
    from tests.test_model.tts_ci_config import TTS_CI_PRESETS

    normalized_value = option_value.strip().lower()
    if normalized_value not in TTS_CI_PRESETS:
        allowed = tuple(sorted(TTS_CI_PRESETS))
        raise pytest.UsageError(
            f"Unsupported value for {TTS_CI_MODEL_OPTION}: {option_value!r}. "
            f"Use one of {allowed}."
        )
    return normalized_value


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        if item.path.name != "test_tts_ci.py":
            continue

        stage_markers = tuple(item.iter_markers(name="tts_stage"))
        if len(stage_markers) != 1:
            raise pytest.UsageError(
                "Each test in tests/test_model/test_tts_ci.py must have "
                "exactly one tts_stage marker."
            )

        stage_ids = tuple(str(arg) for arg in stage_markers[0].args)
        if len(stage_ids) != 1 or stage_ids[0] not in TTS_CI_STAGES:
            raise pytest.UsageError(
                "Each tts_stage marker in tests/test_model/test_tts_ci.py "
                f"must provide exactly one valid stage ID from {TTS_CI_STAGES}."
            )

    selected_stage = config.stash.get(SELECTED_TTS_CI_STAGE, TTS_STAGE_ALL)
    if selected_stage == TTS_STAGE_ALL:
        return

    selected_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []
    for item in items:
        if item.path.name != "test_tts_ci.py":
            selected_items.append(item)
            continue

        stage_marker = item.get_closest_marker("tts_stage")
        assert stage_marker is not None
        stage_ids = tuple(str(arg) for arg in stage_marker.args)
        if selected_stage in stage_ids:
            selected_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items
