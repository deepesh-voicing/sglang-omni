"""SeedTTS benchmark for Voicing-TTS with performance and WER metrics.

Voice cloning (Base checkpoints) uses ref_audio / ref_text from the meta file
by default. For CustomVoice checkpoints pass --no-ref-audio --task-type
CustomVoice --voice <speaker>; for VoiceDesign pass --no-ref-audio --task-type
VoiceDesign --instructions <description>.

WER transcription runs against an external OpenAI-compatible
/v1/audio/transcriptions server given by --asr-host / --asr-port; this
repository does not serve ASR models.

Usage:

1. Download the test set:

    python -m benchmarks.dataset.prepare --dataset seedtts

2. Full pipeline (start Voicing-TTS -> generate -> stop Voicing-TTS -> WER on
   the external ASR server):

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model checkpoints/voicing-tts-12hz-1.7b-base \
        --server-config examples/configs/voicing_tts_1_7b.yaml \
        --max-concurrency 16 \
        --port 8000 --asr-port 8001

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model checkpoints/voicing-tts-12hz-1.7b-customvoice \
        --server-config examples/configs/voicing_tts_1_7b_customvoice.yaml \
        --no-ref-audio --task-type CustomVoice --voice Vivian \
        --max-concurrency 16 \
        --port 8000 --asr-port 8001

3. For CI settings, separate the generate and transcribe phases into two runs.

Usage (CI):

    # Generate audio only

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --generate-only \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --max-concurrency 16 \
        --output-dir results/voicing_tts_en \
        --model checkpoints/voicing-tts-12hz-1.7b-base \
        --server-config examples/configs/voicing_tts_1_7b.yaml \
        --port 8000

    # Transcribe + WER only

    python -m benchmarks.eval.benchmark_tts_seedtts \
        --transcribe-only \
        --meta zhaochenyang20/seed-tts-eval-arrow \
        --model checkpoints/voicing-tts-12hz-1.7b-base \
        --output-dir results/voicing_tts_en \
        --lang en --asr-port 8001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from benchmarks.benchmarker.conditions import (
    ConcurrencyAggregate,
    RepeatSpeedSummary,
    add_fingerprint_argument,
    aggregate_repeats,
    collect_run_fingerprint,
    warn_if_tail_percentile_is_thin,
)
from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.fingerprint import BenchmarkFingerprint
from benchmarks.benchmarker.runner import (
    BenchmarkRunner,
    RunConfig,
    SendFn,
    resolve_warmup,
)
from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import (
    build_speed_results,
    compute_speed_metrics,
    print_speed_summary,
)
from benchmarks.tasks.asr import (
    DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    QWEN3_ASR_MODEL_PATH,
)
from benchmarks.tasks.tts import (
    build_base_url,
    make_tts_send_fn,
    run_seedtts_similarity,
    run_seedtts_transcribe,
    run_seedtts_utmos,
    save_generated_audio_metadata,
    save_speed_results,
)
from sglang_omni.admission import QueueFullError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TTS_BENCHMARK_CONCURRENCY = int(os.getenv("TTS_BENCHMARK_CONCURRENCY", "16"))
DEFAULT_TTS_MODEL = "checkpoints/voicing-tts-12hz-1.7b-base"
DEFAULT_ASR_HOST = "127.0.0.1"


@dataclass
class TtsSeedttsBenchmarkConfig:
    model: str
    meta: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    voice: str | None = None
    task_type: str | None = None
    instructions: str | None = None
    voice_clone: bool = True
    ref_format: str = "flat"
    no_ref_text: bool = False
    response_format: str = "wav"
    output_dir: str = "results/tts_seedtts"
    max_samples: int | None = None
    # Note (Yueying Li): skip this many samples before taking max_samples — lets N concurrent
    # clients replay DISJOINT dataset shards (offset i*max_samples) so shared
    # radix/fingerprint caches don't inflate multi-client throughput.
    sample_offset: int = 0
    max_new_tokens: int | None = 2048
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    # Note (Shulei He): Fraction of requests sent with subtalker_dosample=True (rest use False),
    # deterministically alternated by sample index so runs are reproducible.
    # None leaves subtalker_dosample unset, i.e. the server-side default applies.
    subtalker_dosample_ratio: float | None = None
    warmup: int | None = None
    concurrency: int = DEFAULT_TTS_BENCHMARK_CONCURRENCY
    request_rate: float = float("inf")
    stream: bool = False
    initial_codec_chunk_frames: int | None = None
    disable_tqdm: bool = False
    max_running_requests: int = 64
    max_queued_requests: int | None = None
    overshoot_duration_s: float = 10.0
    cuda_graph_max_bs: int = 64
    # note (luojiaxuan): optional sglang-omni pipeline config yaml forwarded
    # to the managed TTS server as --config, e.g.
    # examples/configs/voicing_tts_1_7b.yaml.
    server_config: str | None = None
    quantization: str | None = None
    lang: str = "en"
    device: str = "cuda:0"
    similarity_checkpoint: str | None = None
    asr_model_path: str = QWEN3_ASR_MODEL_PATH
    asr_concurrency: int = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
    environment_fingerprint: BenchmarkFingerprint | None = None


def _build_generation_kwargs(config: TtsSeedttsBenchmarkConfig) -> dict:
    generation_kwargs: dict = {}
    if config.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = config.max_new_tokens
    if config.temperature is not None:
        generation_kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        generation_kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        generation_kwargs["top_k"] = config.top_k
    if config.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = config.repetition_penalty
    if config.seed is not None:
        generation_kwargs["seed"] = config.seed
    return generation_kwargs


def _resolve_warmup(config: TtsSeedttsBenchmarkConfig) -> int:
    return resolve_warmup(config.warmup, config.concurrency)


def _subtalker_dosample_flags(
    samples: list[SampleInput],
    ratio: float,
) -> dict[str, bool]:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"--subtalker-dosample-ratio must be in [0, 1], got {ratio}")
    flags: dict[str, bool] = {}
    for index, sample in enumerate(samples):
        dosample = math.floor((index + 1) * ratio) > math.floor(index * ratio)
        flags[sample.sample_id] = dosample
    return flags


def _build_results_config(
    config: TtsSeedttsBenchmarkConfig,
    *,
    base_url: str,
) -> dict:
    recorded = {
        "model": config.model,
        "base_url": base_url,
        "meta": config.meta,
        "voice_clone": config.voice_clone,
        "ref_format": config.ref_format,
        "no_ref_text": config.no_ref_text,
        "response_format": config.response_format,
        "voice": config.voice,
        "task_type": config.task_type,
        "instructions": config.instructions,
        "stream": config.stream,
        "max_samples": config.max_samples,
        "sample_offset": config.sample_offset,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "seed": config.seed,
        "subtalker_dosample_ratio": config.subtalker_dosample_ratio,
        "warmup": _resolve_warmup(config),
        "concurrency": config.concurrency,
        "request_rate": config.request_rate,
        "initial_codec_chunk_frames": config.initial_codec_chunk_frames,
        "max_running_requests": config.max_running_requests,
        "max_queued_requests": config.max_queued_requests,
        "overshoot_duration_s": config.overshoot_duration_s,
        "cuda_graph_max_bs": config.cuda_graph_max_bs,
        "server_config": config.server_config,
        "quantization": config.quantization,
    }
    if config.environment_fingerprint is None:
        return recorded
    return {
        **recorded,
        "environment_fingerprint": config.environment_fingerprint,
    }


def _load_benchmark_samples(config: TtsSeedttsBenchmarkConfig) -> list[SampleInput]:
    # Note (Jiaxin Deng): a negative offset would silently slice from the end
    # instead of skipping the first N, contaminating the shard it claims to take.
    if config.sample_offset < 0:
        raise ValueError(
            f"--sample-offset must be non-negative, got {config.sample_offset}"
        )
    if config.sample_offset:
        head = config.sample_offset + (config.max_samples or 0)
        return load_seedtts_samples(
            config.meta, head if config.max_samples else None, split=config.lang
        )[config.sample_offset :]
    return load_seedtts_samples(config.meta, config.max_samples, split=config.lang)


def _make_subtalker_dosample_send_fn(
    config: TtsSeedttsBenchmarkConfig,
    api_url: str,
    common_send_kwargs: dict,
    generation_kwargs: dict,
    samples: list[SampleInput],
) -> SendFn:
    # Note (Shulei He): alternate Voicing-TTS subtalker_dosample per request for
    # mixed sampled/greedy predictor traffic.
    dosample_by_id = _subtalker_dosample_flags(samples, config.subtalker_dosample_ratio)
    send_fn_true = make_tts_send_fn(
        config.model,
        api_url,
        **common_send_kwargs,
        **generation_kwargs,
        stage_params={"tts_engine": {"subtalker_dosample": True}},
    )
    send_fn_false = make_tts_send_fn(
        config.model,
        api_url,
        **common_send_kwargs,
        **generation_kwargs,
        stage_params={"tts_engine": {"subtalker_dosample": False}},
    )
    send_fn_by_id = {
        sample_id: (send_fn_true if dosample else send_fn_false)
        for sample_id, dosample in dosample_by_id.items()
    }

    async def send_fn(session, sample):
        return await send_fn_by_id[sample.sample_id](session, sample)

    return send_fn


async def run_tts_seedtts_benchmark(
    config: TtsSeedttsBenchmarkConfig,
    *,
    samples: list[SampleInput] | None = None,
    save_audio: bool = True,
) -> dict:
    """Generate audio and measure speed.

    Saves audio by default so the transcribe phase can reuse it.
    """
    base_url = build_base_url(config)
    api_url = f"{base_url}/v1/audio/speech"
    if samples is None:
        samples = _load_benchmark_samples(config)
    logger.info(f"Prepared {len(samples)} requests (offset {config.sample_offset})")

    save_audio_dir = None
    if save_audio:
        save_audio_dir = os.path.abspath(os.path.join(config.output_dir, "audio"))
        os.makedirs(save_audio_dir, exist_ok=True)
    else:
        os.makedirs(config.output_dir, exist_ok=True)

    generation_kwargs = _build_generation_kwargs(config)
    common_send_kwargs = dict(
        response_format=config.response_format,
        stream=config.stream,
        initial_codec_chunk_frames=config.initial_codec_chunk_frames,
        no_ref_audio=not config.voice_clone,
        ref_format=config.ref_format,
        no_ref_text=config.no_ref_text,
        voice=config.voice,
        task_type=config.task_type,
        instructions=config.instructions,
        save_audio_dir=save_audio_dir,
    )
    if config.subtalker_dosample_ratio is None:
        send_fn = make_tts_send_fn(
            config.model, api_url, **common_send_kwargs, **generation_kwargs
        )
    else:
        send_fn = _make_subtalker_dosample_send_fn(
            config, api_url, common_send_kwargs, generation_kwargs, samples
        )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.concurrency,
            request_rate=config.request_rate,
            warmup=_resolve_warmup(config),
            disable_tqdm=config.disable_tqdm,
        )
    )
    outputs = await runner.run(samples, send_fn)
    warn_if_tail_percentile_is_thin(len(outputs))

    metrics = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    results_config = _build_results_config(config, base_url=base_url)
    benchmark_results = build_speed_results(outputs, metrics, results_config)
    save_speed_results(outputs, metrics, results_config, config.output_dir)
    save_generated_audio_metadata(outputs, samples, config.output_dir)
    return benchmark_results


def run_tts_seedtts_transcribe(
    config: TtsSeedttsBenchmarkConfig,
    *,
    asr_router_port: int,
    asr_host: str = DEFAULT_ASR_HOST,
) -> dict:
    """Transcribe saved audio on the external ASR server and compute WER + ASR speed."""
    generation_mode = "streaming-audio" if config.stream else "non-streaming"
    wer_config = {
        "model": config.model,
        "tts_model": config.model,
        "asr_model": config.asr_model_path,
        "meta": config.meta,
        "voice_clone": config.voice_clone,
        "ref_format": config.ref_format,
        "no_ref_text": config.no_ref_text,
        "response_format": config.response_format,
        "voice": config.voice,
        "task_type": config.task_type,
        "instructions": config.instructions,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "seed": config.seed,
        "max_samples": config.max_samples,
        "stream": config.stream,
        "initial_codec_chunk_frames": config.initial_codec_chunk_frames,
        "concurrency": config.concurrency,
        "asr_concurrency": config.asr_concurrency,
        "quantization": config.quantization,
    }
    return run_seedtts_transcribe(
        config,
        wer_config=wer_config,
        generation_mode=generation_mode,
        asr_router_port=asr_router_port,
        asr_host=asr_host,
    )


def _config_from_args(args: argparse.Namespace) -> TtsSeedttsBenchmarkConfig:
    voice_clone = not args.no_ref_audio
    response_format = "pcm" if args.stream else args.response_format
    return TtsSeedttsBenchmarkConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        meta=args.meta,
        voice=args.voice,
        task_type=args.task_type,
        instructions=args.instructions,
        voice_clone=voice_clone,
        ref_format=args.ref_format,
        no_ref_text=args.no_ref_text,
        response_format=response_format,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        sample_offset=args.sample_offset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        subtalker_dosample_ratio=args.subtalker_dosample_ratio,
        warmup=args.warmup,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        stream=args.stream,
        initial_codec_chunk_frames=args.initial_codec_chunk_frames,
        disable_tqdm=args.disable_tqdm,
        max_running_requests=args.max_running_requests,
        max_queued_requests=args.max_queued_requests,
        overshoot_duration_s=args.overshoot_duration_s,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        server_config=args.server_config,
        quantization=args.quantization,
        lang=args.lang,
        device=args.device,
        similarity_checkpoint=args.similarity_checkpoint,
        asr_model_path=args.asr_model_path,
        asr_concurrency=args.asr_concurrency,
    )


def _parse_concurrencies(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(
            "concurrencies must be a non-empty comma-separated list"
        )
    values: list[int] = []
    for token in tokens:
        try:
            parsed = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid concurrency {token!r}") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("concurrency must be > 0")
        values.append(parsed)
    return values


@dataclass(frozen=True)
class SustainedOvershootPlan:
    capacity: int
    request_rate: float
    duration_s: float
    request_count: int
    concurrency: int
    overshoot_ratio: float


def plan_sustained_overshoot(
    *,
    max_running_requests: int,
    max_queued_requests: int,
    duration_s: float,
    request_rate: float | None = None,
    overshoot_factor: float = 2.0,
) -> SustainedOvershootPlan:
    """Open-loop arrivals above running + queued for duration_s."""
    if max_running_requests < 1 or max_queued_requests < 1:
        raise ValueError("max_running_requests and max_queued_requests must be >= 1")
    if duration_s <= 0 or overshoot_factor <= 1:
        raise ValueError("overshoot duration_s must be positive and factor > 1")
    capacity = max_running_requests + max_queued_requests
    rate = overshoot_factor * capacity if request_rate is None else float(request_rate)
    if rate <= capacity:
        raise ValueError(
            "sustained overshoot requires request_rate > admission capacity "
            f"({rate} <= {capacity})"
        )
    return SustainedOvershootPlan(
        capacity=capacity,
        request_rate=rate,
        duration_s=float(duration_s),
        request_count=max(1, math.ceil(rate * duration_s)),
        concurrency=0,
        overshoot_ratio=rate / capacity,
    )


def expand_samples_for_overshoot(
    samples: list[SampleInput], request_count: int
) -> list[SampleInput]:
    if not samples or request_count < 1:
        raise ValueError("sustained overshoot requires samples and request_count >= 1")
    n = len(samples)
    expanded: list[SampleInput] = []
    for index in range(request_count):
        source = samples[index % n]
        expanded.append(replace(source, sample_id=f"{source.sample_id}#{index}"))
    return expanded


def classify_overshoot_outcomes(records: list[Any]) -> dict[str, Any]:
    success_ttfa: list[float] = []
    reject_latency: list[float] = []
    success = queue_full = other_failed = 0
    for record in records:
        if isinstance(record, RequestResult):
            ok, error, ttfa, latency = (
                record.is_success,
                record.error,
                record.audio_ttfp_s,
                record.latency_s,
            )
        else:
            ok = bool(record.get("is_success"))
            error = record.get("error")
            ttfa = record.get("audio_ttfp_s")
            latency = float(record.get("latency_s") or 0.0)
        if ok:
            success += 1
            if ttfa is not None:
                success_ttfa.append(float(ttfa))
        elif QueueFullError.matches(error):
            queue_full += 1
            reject_latency.append(float(latency))
        else:
            other_failed += 1
    return {
        "total_requests": len(records),
        "success": success,
        "queue_full": queue_full,
        "other_failed": other_failed,
        "success_ttfa_p95_s": _percentile(success_ttfa, 95),
        "queue_full_latency_p95_s": _percentile(reject_latency, 95),
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[rank]


async def run_tts_concurrency_sweep(
    config: TtsSeedttsBenchmarkConfig,
    concurrencies: list[int],
    *,
    repeats: int = 1,
) -> dict[str, Any]:
    """Run generate-only across concurrencies and optional repeats."""
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    rows: list[ConcurrencyAggregate] = []
    for concurrency in concurrencies:
        repeat_summaries: list[RepeatSpeedSummary] = []
        for repeat_index in range(1, repeats + 1):
            if repeats == 1:
                point_name = f"c{concurrency}"
            else:
                point_name = f"c{concurrency}_r{repeat_index}"
            point_output_dir = os.path.join(config.output_dir, point_name)
            point = replace(
                config,
                concurrency=concurrency,
                output_dir=point_output_dir,
            )
            print(f"[conc={concurrency} repeat={repeat_index}/{repeats}] generate pass")
            results = await run_tts_seedtts_benchmark(point)
            summary = results["summary"]
            repeat_summaries.append(
                {
                    "repeat": repeat_index,
                    "output_dir": point_output_dir,
                    **summary,
                    "warmup": _resolve_warmup(point),
                }
            )
            print_speed_summary(summary, config.model, concurrency=concurrency)
            print(
                f"  success={summary.get('completed_requests')} "
                f"failed={summary.get('failed_requests')} "
                f"latency_p95={summary.get('latency_p95_s')} "
                f"ttfa_p95={summary.get('audio_ttfp_p95_s')}"
            )
        rows.append(aggregate_repeats(concurrency, repeat_summaries))

    payload = {
        "config": _build_results_config(config, base_url=build_base_url(config)),
        "concurrencies": concurrencies,
        "repeats": repeats,
        "rows": rows,
    }
    out_path = os.path.join(config.output_dir, "concurrency_sweep.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info(f"Wrote concurrency sweep to {out_path}")
    return payload


async def run_tts_sustained_overshoot(
    config: TtsSeedttsBenchmarkConfig,
) -> dict[str, Any]:
    """Hold offered load above admission capacity instead of stepping concurrency."""
    if config.max_queued_requests is None:
        raise ValueError("--sustained-overshoot requires --max-queued-requests")
    request_rate = None if config.request_rate == float("inf") else config.request_rate
    plan = plan_sustained_overshoot(
        max_running_requests=config.max_running_requests,
        max_queued_requests=config.max_queued_requests,
        duration_s=config.overshoot_duration_s,
        request_rate=request_rate,
    )
    point = replace(
        config,
        concurrency=plan.concurrency,
        request_rate=plan.request_rate,
        output_dir=os.path.join(config.output_dir, "overshoot"),
    )
    print(
        f"[overshoot] open-loop rate={plan.request_rate:.3f}/s "
        f"capacity={plan.capacity} duration={plan.duration_s}s "
        f"n={plan.request_count}"
    )
    corpus = _load_benchmark_samples(config)
    samples = expand_samples_for_overshoot(corpus, plan.request_count)
    results = await run_tts_seedtts_benchmark(point, samples=samples, save_audio=False)
    outcomes = classify_overshoot_outcomes(results["per_request"])
    payload = {
        "plan": asdict(plan),
        "outcomes": outcomes,
        "summary": results["summary"],
        "config": results["config"],
    }
    out_path = os.path.join(point.output_dir, "sustained_overshoot.json")
    os.makedirs(point.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info(f"Wrote sustained overshoot to {out_path}")
    print_speed_summary(results["summary"], config.model, concurrency=plan.concurrency)
    print(
        f"  success={outcomes['success']} queue_full={outcomes['queue_full']} "
        f"other_failed={outcomes['other_failed']} "
        f"ttfa_p95={outcomes['success_ttfa_p95_s']} "
        f"reject_p95={outcomes['queue_full_latency_p95_s']}"
    )
    return payload


async def benchmark(config: TtsSeedttsBenchmarkConfig) -> dict:
    results = await run_tts_seedtts_benchmark(config)
    print_speed_summary(
        results["summary"], config.model, concurrency=config.concurrency
    )
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SeedTTS benchmark for Voicing-TTS.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TTS_MODEL,
        help=(
            "Model name for the API request and --model-path of the managed "
            f"server (default: {DEFAULT_TTS_MODEL})."
        ),
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help=(
            "Built-in speaker name for Voicing-TTS CustomVoice checkpoints "
            "(e.g. 'Vivian'); combine with --no-ref-audio. Base checkpoints "
            "take the speaker from ref_audio in the meta file."
        ),
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Voicing-TTS task type: Base, CustomVoice, or VoiceDesign.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Style or VoiceDesign voice-description instructions.",
    )
    parser.add_argument(
        "--meta",
        "--testset",
        dest="meta",
        type=str,
        default="zhaochenyang20/seed-tts-eval-arrow",
        help="HuggingFace Arrow/Parquet dataset repo id or local meta.lst path.",
    )
    parser.add_argument(
        "--no-ref-audio",
        dest="no_ref_audio",
        action="store_true",
        help="Skip ref audio/text from testset (TTS without voice cloning).",
    )
    parser.add_argument(
        "--no-ref-text",
        dest="no_ref_text",
        action="store_true",
        help=(
            "Keep ref_audio (voice cloning) but drop ref_text/references[].text "
            "from the request, for cross-lingual-style runs. Ignored when "
            "--no-ref-audio is also set."
        ),
    )
    parser.add_argument(
        "--ref-format",
        choices=["flat", "references"],
        default="flat",
        help=(
            "Reference payload shape for voice cloning. The default 'flat' sends "
            "ref_audio/ref_text; 'references' sends "
            "references=[{audio_path, text}]."
        ),
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        help=(
            "Requested audio payload format. Streaming always sends "
            "response_format=pcm."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="results/tts_seedtts")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Per-request sampler seed for reproducible generation.",
    )
    parser.add_argument(
        "--subtalker-dosample-ratio",
        type=float,
        default=None,
        help=(
            "Optional fraction of requests "
            "sent with subtalker_dosample=True (the rest use False), "
            "deterministically interleaved by sample index. 1.0 = all "
            "sampled, 0.0 = all greedy, 0.5 = alternating mixed traffic. "
            "Omit to leave subtalker_dosample unset (server-side default)."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warmup requests; defaults to the configured concurrency.",
    )
    parser.add_argument(
        "--concurrency",
        "--max-concurrency",
        dest="concurrency",
        type=int,
        default=DEFAULT_TTS_BENCHMARK_CONCURRENCY,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--concurrencies",
        type=_parse_concurrencies,
        default=None,
        help="Comma-separated concurrency levels to sweep (requires --generate-only).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Measured passes per concurrency. Requires --generate-only when above 1.",
    )
    add_fingerprint_argument(parser)
    parser.add_argument(
        "--sustained-overshoot",
        action="store_true",
        help="Open-loop soak above running+queued (needs --generate-only and --max-queued-requests).",
    )
    parser.add_argument(
        "--overshoot-duration-s",
        type=float,
        default=10.0,
        help="Wall-clock seconds to keep offering overshoot load.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests/s (inf = burst). Soak defaults to 2x capacity if omitted.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming for TTS generation.",
    )
    parser.add_argument(
        "--initial-codec-chunk-frames",
        type=int,
        default=None,
        help=(
            "Optional first streaming vocoder chunk size in codec frames; "
            "later chunks use the steady chunk size."
        ),
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Legacy flag kept for backward compatibility. The unified "
        "benchmark always saves generated WAVs so the transcribe phase can "
        "reuse them; passing this flag is a no-op.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="Dataset split and ASR language (transcribe phase).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for speaker-similarity and UTMOS scoring.",
    )
    parser.add_argument(
        "--asr-host",
        type=str,
        default=DEFAULT_ASR_HOST,
        help="Host of the external OpenAI-compatible ASR server used for WER.",
    )
    parser.add_argument(
        "--asr-port",
        type=int,
        default=None,
        help=(
            "Port of the external OpenAI-compatible ASR server "
            "(/v1/audio/transcriptions) used for WER. Required unless "
            "--generate-only, --similarity-only, or --utmos-only is set."
        ),
    )
    parser.add_argument(
        "--asr-model-path",
        type=str,
        default=QWEN3_ASR_MODEL_PATH,
        help=(
            "Model name sent to the external ASR server. Defaults to "
            f"{QWEN3_ASR_MODEL_PATH}; names containing 'whisper' are sent in "
            "30 s chunks."
        ),
    )
    parser.add_argument(
        "--asr-concurrency",
        type=int,
        default=DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
        help="Concurrent transcription requests during WER evaluation.",
    )
    parser.add_argument(
        "--similarity-checkpoint",
        type=str,
        default=None,
        help="Optional path to a custom fine-tuned WavLM checkpoint. "
        "If omitted, the official weights are downloaded into a local cache "
        "directory (override the cache root with SEEDTTS_SIM_CACHE_DIR).",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds to wait for server readiness.",
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=64,
        help=(
            "SGLang generation stage max_running_requests for the server "
            "started by this benchmark. Recommended to keep equal to "
            "--cuda-graph-max-bs. Defaults to 64."
        ),
    )
    parser.add_argument(
        "--max-queued-requests",
        type=int,
        default=None,
        help=(
            "SGLang generation stage max_queued_requests for the managed "
            "server. Omit to leave the pipeline default."
        ),
    )
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=64,
        help=(
            "SGLang generation stage cuda_graph_max_bs for the server "
            "started by this benchmark. Recommended to keep equal to "
            "--max-running-requests. Defaults to 64."
        ),
    )
    parser.add_argument(
        "--server-config",
        type=str,
        default=None,
        help=(
            "Optional sglang-omni pipeline config yaml passed to the managed "
            "TTS server as --config (e.g. "
            "examples/configs/voicing_tts_1_7b.yaml). Ignored with "
            "--use-existing-server."
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help=(
            "SGLang quantization mode (e.g. fp8) for the TTS generation stage "
            "of the server started by this benchmark. Defaults to none (bf16)."
        ),
    )
    parser.add_argument(
        "--skip-gpu-cleanup",
        action="store_true",
        help=(
            "Do not run ensure_gpus_idle after stopping a server. Use when "
            "running multiple benchmark processes in parallel on different "
            "GPUs; combine with CUDA_VISIBLE_DEVICES per worker and clean up "
            "each GPU once after the worker finishes."
        ),
    )
    parser.add_argument(
        "--use-existing-server",
        action="store_true",
        help=(
            "Do not start or stop a server; send requests to the configured "
            "--base-url or --host/--port instead."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Only synthesize audio and measure speed; skip WER transcription.",
    )
    mode.add_argument(
        "--transcribe-only",
        action="store_true",
        help=(
            "Only run ASR transcription and WER on existing output-dir, "
            "against the external ASR server."
        ),
    )
    mode.add_argument(
        "--similarity-only",
        action="store_true",
        help="Only run speaker similarity on existing output-dir.",
    )
    mode.add_argument(
        "--utmos-only",
        action="store_true",
        help="Only run UTMOS MOS scoring on existing output-dir.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    is_sweep = args.concurrencies is not None or args.repeats > 1
    needs_asr = not (args.generate_only or args.similarity_only or args.utmos_only)
    if (
        args.initial_codec_chunk_frames is not None
        and args.initial_codec_chunk_frames < 0
    ):
        parser.error("--initial-codec-chunk-frames must be non-negative")
    elif args.max_running_requests <= 0:
        parser.error("--max-running-requests must be positive")
    elif args.max_queued_requests is not None and args.max_queued_requests < 1:
        parser.error("--max-queued-requests must be >= 1")
    elif args.cuda_graph_max_bs <= 0:
        parser.error("--cuda-graph-max-bs must be positive")
    elif args.repeats < 1:
        parser.error("--repeats must be positive")
    elif is_sweep and not args.generate_only:
        parser.error("--concurrencies and --repeats require --generate-only")
    elif args.sustained_overshoot and not args.generate_only:
        parser.error("--sustained-overshoot currently requires --generate-only")
    elif args.sustained_overshoot and is_sweep:
        parser.error(
            "--sustained-overshoot cannot be combined with --concurrencies or --repeats"
        )
    elif args.sustained_overshoot and args.max_queued_requests is None:
        parser.error("--sustained-overshoot requires --max-queued-requests")
    elif args.overshoot_duration_s <= 0:
        parser.error("--overshoot-duration-s must be positive")
    elif args.use_existing_server and not (args.generate_only or args.transcribe_only):
        parser.error(
            "--use-existing-server currently requires --generate-only or "
            "--transcribe-only"
        )
    elif needs_asr and args.asr_port is None:
        parser.error(
            "WER needs an external OpenAI-compatible ASR server; pass --asr-port "
            "(and --asr-host if it is not local), or use --generate-only"
        )
    elif args.sustained_overshoot and args.request_rate != float("inf"):
        try:
            plan_sustained_overshoot(
                max_running_requests=args.max_running_requests,
                max_queued_requests=args.max_queued_requests,
                duration_s=args.overshoot_duration_s,
                request_rate=args.request_rate,
            )
        except ValueError as exc:
            parser.error(str(exc))
    else:
        pass


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    config = _config_from_args(args)

    if args.save_audio:
        logger.info("--save-audio is a no-op: the unified benchmark always saves WAVs.")
    else:
        pass

    if args.similarity_only:
        run_seedtts_similarity(config)
        return
    elif args.utmos_only:
        run_seedtts_utmos(config, log_per_sample=True)
        return
    elif args.transcribe_only:
        run_tts_seedtts_transcribe(
            config, asr_router_port=args.asr_port, asr_host=args.asr_host
        )
        return
    else:
        pass

    async def _run_generate() -> None:
        if args.fingerprint:
            config.environment_fingerprint = collect_run_fingerprint(
                build_base_url(config)
            )
        else:
            config.environment_fingerprint = None
        is_sweep = args.concurrencies is not None or args.repeats > 1
        if is_sweep:
            if args.concurrencies is None:
                concurrencies = [config.concurrency]
            else:
                concurrencies = args.concurrencies
            await run_tts_concurrency_sweep(config, concurrencies, repeats=args.repeats)
        elif args.sustained_overshoot:
            await run_tts_sustained_overshoot(config)
        else:
            await benchmark(config)

    if args.use_existing_server:
        asyncio.run(_run_generate())
    else:
        with managed_omni_server(
            model_path=config.model,
            port=config.port,
            host=config.host,
            server_config=config.server_config,
            max_running_requests=config.max_running_requests,
            max_queued_requests=config.max_queued_requests,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            quantization=config.quantization,
            log_file=Path(config.output_dir) / "server_logs" / "tts_server.log",
            timeout=args.server_timeout,
            wait_for_gpu_release=not args.skip_gpu_cleanup,
        ):
            asyncio.run(_run_generate())

    if args.generate_only:
        return
    else:
        pass

    run_tts_seedtts_transcribe(
        config, asr_router_port=args.asr_port, asr_host=args.asr_host
    )


if __name__ == "__main__":
    main()
