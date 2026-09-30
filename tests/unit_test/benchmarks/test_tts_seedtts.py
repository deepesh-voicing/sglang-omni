# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark entry-point: server lifecycle, external ASR, WER filter."""

import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import pytest
import requests

from benchmarks.eval import benchmark_tts_seedtts as tts
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks import asr
from tests.utils import WER_ASR_CONCURRENCY, assert_wer_partitioned


def test_evaluation_releases_tts_server_before_external_asr(monkeypatch):
    model = "checkpoints/voicing-tts-12hz-1.7b-base"
    events = []
    servers = []

    @contextmanager
    def server(**kwargs):
        servers.append(kwargs)
        events.append("start")
        yield
        events.append("stop")

    async def generate(config):
        assert config.port == 18280
        assert config.max_samples == 2
        assert config.model == model
        events.append("generate")

    def transcribe(config, **kwargs):
        assert kwargs["asr_router_port"] == 30001
        assert kwargs["asr_host"] == "asr.internal"
        events.append("transcribe")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            model,
            "--port",
            "18280",
            "--max-samples",
            "2",
            "--asr-host",
            "asr.internal",
            "--asr-port",
            "30001",
        ],
    )
    monkeypatch.setattr(tts, "managed_omni_server", server)
    monkeypatch.setattr(tts, "benchmark", generate)
    monkeypatch.setattr(tts, "run_tts_seedtts_transcribe", transcribe)
    tts.main()

    # WER runs on an external ASR server, so only the TTS server is managed.
    assert events == ["start", "generate", "stop", "transcribe"]
    assert len(servers) == 1
    assert servers[0]["model_path"] == model
    assert servers[0]["max_running_requests"] == 64
    assert servers[0]["cuda_graph_max_bs"] == 64


def test_filtered_wer_mean_keeps_exactly_50_percent_and_excludes_failures():
    metrics = calculate_wer_metrics(
        [
            SampleOutput(is_success=True, wer=0, hits=10),
            SampleOutput(is_success=True, wer=0.5, hits=1, deletions=1),
            SampleOutput(is_success=True, wer=0.75, hits=1, deletions=3),
            SampleOutput(is_success=False),
        ],
        "en",
    )
    assert metrics["wer_below_50_per_sample_mean"] == 0.25
    assert metrics["wer_below_50_corpus"] == pytest.approx(1 / 12)
    assert metrics["n_above_50_pct_wer"] == 1
    assert metrics["evaluated"] == 3
    assert metrics["skipped"] == 1


def test_wer_fanout_preserves_all_twenty_samples_at_long_audio_admission_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # note (wenyao): routing can send every request to one four-slot worker.
    slots = threading.BoundedSemaphore(4)
    cohort = threading.Barrier(min(WER_ASR_CONCURRENCY, 20))
    uploaded: list[str] = []

    def post(
        url: str, *, files: dict[str, tuple[str, BinaryIO, str]], **kwargs: object
    ) -> requests.Response:
        admitted = slots.acquire(blocking=False)
        try:
            # note (wenyao): all requests attempt admission before slots reopen.
            cohort.wait(timeout=5)
            uploaded.append(files["file"][0])
            response = requests.Response()
            response.url = url
            response.status_code = 200 if admitted else 503
            response._content = json.dumps(  # noqa: leading-underscore  # upstream name
                {"text": "hello world"}
                if admitted
                else {
                    "detail": "Too many long-audio transcriptions in flight "
                    "(limit 4); retry later"
                }
            ).encode()
            return response
        finally:
            if admitted:
                slots.release()

    monkeypatch.setattr(asr.requests, "post", post)
    records: list[dict[str, str | bool | int]] = []
    for index in range(20):
        path = tmp_path / f"sample-{index}.wav"
        path.write_bytes(b"saved audio for mocked transcription service")
        records.append(
            {
                "sample_id": f"sample-{index}",
                "raw_response": "hello world",
                "is_success": True,
                "wav_path": str(path),
                "audio_duration_s": 31,
            }
        )
    result = asr.compute_text_audio_consistency_from_records(
        records,
        "en",
        "cuda:0",
        asr_router_port=12345,
        asr_concurrency=WER_ASR_CONCURRENCY,
    )

    assert len(uploaded) == len(set(uploaded)) == 20
    assert result["summary"]["evaluated"] == 20
    assert result["summary"]["skipped"] == 0
    assert_wer_partitioned(result, max_wer_below_50_corpus=0, max_n_above_50=0)
    assert asr.DEFAULT_ASR_TRANSCRIBE_CONCURRENCY == 32
