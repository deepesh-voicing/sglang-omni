# SPDX-License-Identifier: Apache-2.0
"""Model presets and thresholds for TTS CI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from tests.utils import apply_mos_slack, apply_slack, apply_wer_slack


@dataclass(frozen=True)
class TtsCiModelPreset:
    model_path: str
    ref_format: Literal["flat", "references"] = "flat"
    worker_extra_args: str = ""
    startup_timeout: int = 180
    gate_thresholds: bool = True
    num_gpus_per_worker: int = 1
    # note (luojiaxuan): a checkpoint that serves named voices takes the text
    # alone, so the reference fields above go unused and the reference-based
    # similarity stage has nothing to score.
    voice: str | None = None
    voice_clone: bool = True


@dataclass(frozen=True)
class TtsCiThresholdPreset:
    non_stream_speed: dict[int, dict[str, float]]
    stream_speed: dict[int, dict[str, float]]
    wer_corpus: float
    stream_wer_corpus: float
    similarity_mean_min: float
    utmos_mean_min: float
    # Note: (Jiaxin Deng) False while the values are seeds rather than
    # worst-of-N observations from the CI runner. A seed can be far off for a
    # topology it was not measured on, so gating on one fails builds for no
    # reason or waves regressions through; a contract test refuses that pair.
    calibrated: bool = True


@dataclass(frozen=True)
class TtsCiPreset:
    model: TtsCiModelPreset
    thresholds: TtsCiThresholdPreset


# Slack factors applied to P95 reference values to derive CI thresholds.
# Higher-is-better metrics: threshold = P95 * slack_higher.
# Lower-is-better metrics: threshold = P95 * slack_lower.
THRESHOLD_SLACK_HIGHER = 0.75
THRESHOLD_SLACK_LOWER = 1.25

DEFAULT_TTS_CI_MODEL = "voicing-tts"
# Converted checkpoints: python -m sglang_omni.models.voicing_tts.convert_checkpoint
# Qwen/Qwen3-TTS-12Hz-1.7B-Base checkpoints/voicing-tts-12hz-1.7b-base (and the
# CustomVoice checkpoint likewise).
VOICING_TTS_MODEL_PATH_ENV = "VOICING_TTS_MODEL_PATH"
VOICING_TTS_CUSTOM_VOICE_MODEL_PATH_ENV = "VOICING_TTS_CUSTOM_VOICE_MODEL_PATH"
VOICING_TTS_MODEL_PATH = os.environ.get(
    VOICING_TTS_MODEL_PATH_ENV, "checkpoints/voicing-tts-12hz-1.7b-base"
)
VOICING_TTS_CUSTOM_VOICE_MODEL_PATH = os.environ.get(
    VOICING_TTS_CUSTOM_VOICE_MODEL_PATH_ENV,
    "checkpoints/voicing-tts-12hz-1.7b-customvoice",
)

# Voicing-TTS 1.7B Base, gated as a single instance per GPU.
#
# Note: (wenyao) measured on the CI host with the same 1.7B Base weights before
# the checkpoint conversion, lane 2,3 pinned cpuset (16-31,80-95), worst-of-5
# clean rounds with destructive rejection (run
# .tune-runs/20260830T024753Z_tts_combined). Raw pre-slack references only; the
# CI slack calculation is unchanged.
VOICING_TTS_VC_WER_MAX_CORPUS = 0.0114
VOICING_TTS_VC_WER_CORPUS_THRESHOLD = apply_wer_slack(VOICING_TTS_VC_WER_MAX_CORPUS)
VOICING_TTS_VC_STREAM_WER_MAX_CORPUS = 0.0109
VOICING_TTS_VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(
    VOICING_TTS_VC_STREAM_WER_MAX_CORPUS
)
VOICING_TTS_VC_SIMILARITY_MEAN_MIN = 69.13817592620849
VOICING_TTS_VC_UTMOS_MEAN_REFERENCE = 4.193
VOICING_TTS_VC_UTMOS_MEAN_MIN = apply_mos_slack(VOICING_TTS_VC_UTMOS_MEAN_REFERENCE)

VOICING_TTS_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 21.083,
        "output_tok_per_req_s": 88.4,
        "latency_mean_s": 0.755,
        "rtf_mean": 0.1859,
    }
}

VOICING_TTS_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 19.526,
        "latency_mean_s": 0.815,
        "rtf_mean": 0.1992,
    }
}

VOICING_TTS_VC_NON_STREAM_THRESHOLDS = apply_slack(
    VOICING_TTS_VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
VOICING_TTS_VC_STREAM_THRESHOLDS = apply_slack(
    VOICING_TTS_VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)

# note (luojiaxuan): docs/cookbook/voicing_tts.md, 1.7B CustomVoice,
# Ryan/English, concurrency 16 on one H200; single runs, so these are
# references to print next to, not worst-of-N observations from the CI host.
VOICING_TTS_CUSTOM_VOICE_NON_STREAM_REFERENCE = {
    16: {
        "throughput_qps": 14.788,
        "latency_mean_s": 1.075,
        "rtf_mean": 0.2335,
    }
}
VOICING_TTS_CUSTOM_VOICE_STREAM_REFERENCE = {
    16: {
        "throughput_qps": 10.098,
        "latency_mean_s": 1.573,
        "rtf_mean": 0.3380,
    }
}
VOICING_TTS_CUSTOM_VOICE_NON_STREAM_THRESHOLDS = apply_slack(
    VOICING_TTS_CUSTOM_VOICE_NON_STREAM_REFERENCE,
    THRESHOLD_SLACK_HIGHER,
    THRESHOLD_SLACK_LOWER,
)
VOICING_TTS_CUSTOM_VOICE_STREAM_THRESHOLDS = apply_slack(
    VOICING_TTS_CUSTOM_VOICE_STREAM_REFERENCE,
    THRESHOLD_SLACK_HIGHER,
    THRESHOLD_SLACK_LOWER,
)
VOICING_TTS_CUSTOM_VOICE_WER_CORPUS_THRESHOLD = apply_wer_slack(0.01608)
VOICING_TTS_CUSTOM_VOICE_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(0.02085)
VOICING_TTS_CUSTOM_VOICE_UTMOS_MEAN_MIN = apply_mos_slack(4.1723)

# Note: (Jiaxin Deng) the shipped defaults colocate every stage in one process,
# so CI splits the vocoder out and measures the tuned point.
VOICING_TTS_WORKER_EXTRA_ARGS = (
    "--vocoder.process vocoder "
    "--tts_engine.gpu_memory_fraction 0.85 "
    "--vocoder.gpu_memory_fraction 0.10"
)
# note (luojiaxuan): a cold Inductor cache compiles the vocoder steady shapes at
# startup, which takes two workers past five minutes.
VOICING_TTS_STARTUP_TIMEOUT_S = 900


TTS_CI_PRESETS: dict[str, TtsCiPreset] = {
    "voicing-tts": TtsCiPreset(
        model=TtsCiModelPreset(
            model_path=VOICING_TTS_MODEL_PATH,
            ref_format="references",
            worker_extra_args=VOICING_TTS_WORKER_EXTRA_ARGS,
            startup_timeout=VOICING_TTS_STARTUP_TIMEOUT_S,
            gate_thresholds=True,
        ),
        thresholds=TtsCiThresholdPreset(
            non_stream_speed=VOICING_TTS_VC_NON_STREAM_THRESHOLDS,
            stream_speed=VOICING_TTS_VC_STREAM_THRESHOLDS,
            wer_corpus=VOICING_TTS_VC_WER_CORPUS_THRESHOLD,
            stream_wer_corpus=VOICING_TTS_VC_STREAM_WER_CORPUS_THRESHOLD,
            similarity_mean_min=VOICING_TTS_VC_SIMILARITY_MEAN_MIN,
            utmos_mean_min=VOICING_TTS_VC_UTMOS_MEAN_MIN,
        ),
    ),
    "voicing-tts-custom-voice": TtsCiPreset(
        model=TtsCiModelPreset(
            model_path=VOICING_TTS_CUSTOM_VOICE_MODEL_PATH,
            voice="Ryan",
            voice_clone=False,
            # note (luojiaxuan): same tuned point as the Base arm, so the two
            # differ only in the checkpoint and the request shape.
            worker_extra_args=(
                "--tts_engine.engine.max_running_requests 64 "
                "--tts_engine.engine.cuda_graph_max_bs 64 "
                "--tts_engine.engine.torch_compile_max_bs 64 "
                f"{VOICING_TTS_WORKER_EXTRA_ARGS}"
            ),
            startup_timeout=VOICING_TTS_STARTUP_TIMEOUT_S,
            gate_thresholds=False,
        ),
        # note (luojiaxuan): printed next to the stage results; this arm gates
        # nothing until it is calibrated on the CI host, which a contract test
        # enforces. The similarity stage skips named voices, so that field is
        # never read here.
        thresholds=TtsCiThresholdPreset(
            non_stream_speed=VOICING_TTS_CUSTOM_VOICE_NON_STREAM_THRESHOLDS,
            stream_speed=VOICING_TTS_CUSTOM_VOICE_STREAM_THRESHOLDS,
            wer_corpus=VOICING_TTS_CUSTOM_VOICE_WER_CORPUS_THRESHOLD,
            stream_wer_corpus=VOICING_TTS_CUSTOM_VOICE_STREAM_WER_CORPUS_THRESHOLD,
            similarity_mean_min=0.0,
            utmos_mean_min=VOICING_TTS_CUSTOM_VOICE_UTMOS_MEAN_MIN,
            calibrated=False,
        ),
    ),
}


def select_tts_ci_preset(model_name: str | None = None) -> tuple[str, TtsCiPreset]:
    selected = model_name or os.environ.get("TTS_CI_MODEL", DEFAULT_TTS_CI_MODEL)
    preset = TTS_CI_PRESETS.get(selected)
    if preset is None:
        allowed = ", ".join(sorted(TTS_CI_PRESETS))
        raise ValueError(
            f"Unsupported TTS_CI_MODEL={selected!r}; expected one of: {allowed}"
        )
    else:
        pass
    return selected, preset
