# SPDX-License-Identifier: Apache-2.0
"""Contracts for comparable benchmark speed measurements."""

from __future__ import annotations

import logging

import pytest

from benchmarks.benchmarker.conditions import (
    SWEEP_METRIC_NAMES,
    aggregate_numbers,
    aggregate_repeats,
    fingerprint_fields,
    sampling_seed_field,
    warn_if_tail_percentile_is_thin,
)


def test_sampling_seed_field_keeps_zero_and_omits_unset() -> None:
    assert sampling_seed_field(0) == {"seed": 0}
    assert sampling_seed_field(None) == {}


def test_aggregate_numbers_empty_and_populated() -> None:
    assert aggregate_numbers([]) == {"mean": None, "min": None, "max": None, "n": 0}
    assert aggregate_numbers([1.0, None, 3.0]) == {
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
        "n": 2,
    }


def test_sweep_aggregate_uses_the_shared_metric_names() -> None:
    row = aggregate_repeats(
        1,
        [
            {
                "repeat": 1,
                "output_dir": "c1",
                "completed_requests": 1,
                "failed_requests": 0,
            }
        ],
    )
    assert row["repeats"] == 1
    assert len(row["per_repeat"]) == 1
    for metric_name in SWEEP_METRIC_NAMES:
        assert set(row[metric_name]) == {"mean", "min", "max", "n"}


def test_sweep_aggregate_records_warmup_and_time_to_first_audio() -> None:
    row = aggregate_repeats(
        1,
        [
            {
                "repeat": 1,
                "output_dir": "c1_r1",
                "completed_requests": 1,
                "failed_requests": 0,
                "warmup": 2,
                "audio_ttfp_mean_s": 0.25,
                "audio_ttfp_p95_s": 0.5,
            },
            {
                "repeat": 2,
                "output_dir": "c1_r2",
                "completed_requests": 1,
                "failed_requests": 0,
                "warmup": 2,
                "audio_ttfp_mean_s": 0.75,
                "audio_ttfp_p95_s": None,
            },
        ],
    )
    assert row["warmup"] == {"mean": 2.0, "min": 2.0, "max": 2.0, "n": 2}
    assert row["audio_ttfp_mean_s"] == {"mean": 0.5, "min": 0.25, "max": 0.75, "n": 2}
    assert row["audio_ttfp_p95_s"] == {"mean": 0.5, "min": 0.5, "max": 0.5, "n": 1}


def test_fingerprint_fields_skip_collection_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_client(model_path: str | None = None) -> dict[str, str]:
        raise AssertionError(model_path)

    def fail_server(base_url: str) -> dict[str, str]:
        raise AssertionError(base_url)

    monkeypatch.setattr(
        "benchmarks.benchmarker.conditions.collect_environment_fingerprint",
        fail_client,
    )
    monkeypatch.setattr(
        "benchmarks.benchmarker.conditions.collect_server_identity",
        fail_server,
    )
    assert fingerprint_fields(False, "http://localhost:8000") == {}


def test_tail_warning_fires_below_one_hundred_samples(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        warn_if_tail_percentile_is_thin(2)
    assert "two slowest of 2 requests" in caplog.text
