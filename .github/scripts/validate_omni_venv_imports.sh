#!/usr/bin/env bash
# Import probe for the Omni CI venv (matches packages exercised in real CI jobs).
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

if [ -z "${OMNI_CI_HOME:-}" ]; then
  echo "OMNI_CI_HOME is not set" >&2
  exit 1
fi

VENV_NAME="$1"
PYTHON="${OMNI_CI_HOME}/${VENV_NAME}/bin/python"

if [ ! -x "${PYTHON}" ]; then
  echo "python not found: ${PYTHON}" >&2
  exit 1
fi

if ! "${PYTHON}" -c "
import av
import torch
import transformers
import sglang
import zhon.hanzi
from whisper.normalizers import EnglishTextNormalizer
from sglang_omni.models.voicing_tts.compat import apply_voicing_tts_transformers_compatibility_patches
apply_voicing_tts_transformers_compatibility_patches()
from sglang_omni.vendor.voicing_tts.inference.voicing_tts_model import VoicingTTSModel
from sglang_omni.vendor.voicing_tts.inference.voicing_tts_tokenizer import VoicingTTSTokenizer
"; then
  echo "::error::${VENV_NAME} import probe failed at ${OMNI_CI_HOME}/${VENV_NAME}" >&2
  exit 1
fi

echo "Import probe ok: ${OMNI_CI_HOME}/${VENV_NAME}"
