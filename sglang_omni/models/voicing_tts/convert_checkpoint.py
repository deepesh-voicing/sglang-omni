# SPDX-License-Identifier: Apache-2.0
"""Copy a Qwen3-TTS checkpoint and rename its model types for the Voicing-TTS loaders."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

CONFIG_VALUE_RENAMES = {
    "Qwen3TTSForConditionalGeneration": "VoicingTTSForConditionalGeneration",
    "Qwen3TTSProcessor": "VoicingTTSProcessor",
    "Qwen3TTSTokenizerV2Model": "VoicingTTSTokenizerV2Model",
    "Qwen2Tokenizer": "VoicingTTSTextTokenizer",
    "Qwen2TokenizerFast": "VoicingTTSTextTokenizer",
    "qwen3_tts": "voicing_tts",
    "qwen3_tts_talker": "voicing_tts_talker",
    "qwen3_tts_talker_code_predictor": "voicing_tts_talker_code_predictor",
    "qwen3_tts_tokenizer_12hz": "voicing_tts_tokenizer_12hz",
}
RENAMED_CONFIG_FILES = (
    "config.json",
    "preprocessor_config.json",
    "speech_tokenizer/config.json",
    "tokenizer_config.json",
)


def rename_config_values(config_value: Any) -> Any:
    if isinstance(config_value, dict):
        return {key: rename_config_values(value) for key, value in config_value.items()}
    elif isinstance(config_value, list):
        return [rename_config_values(value) for value in config_value]
    elif isinstance(config_value, str):
        return CONFIG_VALUE_RENAMES.get(config_value, config_value)
    else:
        return config_value


def convert_checkpoint(source: str, target_dir: Path) -> None:
    if Path(source).is_dir():
        source_dir = Path(source)
    else:
        source_dir = Path(snapshot_download(source))
    shutil.copytree(source_dir, target_dir)
    for config_name in RENAMED_CONFIG_FILES:
        config_path = target_dir / config_name
        config = json.loads(config_path.read_text(encoding="utf-8"))
        renamed_config = rename_config_values(config)
        config_path.write_text(
            json.dumps(renamed_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info(f"Renamed model types in {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", help="Qwen3-TTS checkpoint directory or Hugging Face repo id."
    )
    parser.add_argument(
        "target_dir", type=Path, help="New directory for the Voicing-TTS checkpoint."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert_checkpoint(args.source, args.target_dir)


if __name__ == "__main__":
    main()
else:
    pass
