# 🚀 Installation

Current stable release: **v0.1.6** on [PyPI](https://pypi.org/project/sglang-omni/).

Choose the path for your platform. Docker is recommended for NVIDIA CUDA —
UCX, flash-attn, SGLang, and CUDA are prebuilt.

Voicing-TTS (`sglang_omni.models.voicing_tts`) ships in this repository's
source tree. Install from a checkout of this repository (see
[Install from source](#install-from-source)) to get it; a PyPI release that
predates it does not include the model.

> **Intel GPU (XPU)?** For Intel Arc GPUs, see [Installation — Intel XPU](./installation_xpu.md), which uses [`pyproject_xpu.toml`](../../pyproject_xpu.toml) + the PyTorch XPU wheel index instead of the CUDA-only pins below.

> **Moore Threads GPU (MUSA):** For MUSA, see [Installation — MUSA](./installation_musa.md), which uses [`pyproject_musa.toml`](../../pyproject_musa.toml) on top of a SGLang MUSA environment.

> **Intel CPU?** Also not this page. See [Installation — Intel CPU](./installation_cpu.md), which uses [`pyproject_cpu.toml`](../../pyproject_cpu.toml) + the PyTorch CPU wheel index.

> **Ascend NPU?** See [Installation — Ascend NPU](./installation_npu.md) for the supported software stack, prerequisites, and installation helper.

## 🐳 Option A: Docker (recommended)

**1. Pull the image**

```bash
docker pull hongccc/sglang-omni:dev
```

Only the `dev` tag is published today. It moves with main — pin by digest for reproducible runs:

```bash
docker pull lmsysorg/sglang-omni@sha256:<digest>
```

**2. Run the container**

```bash
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    hongccc/sglang-omni:dev \
    /bin/zsh
```

**3. Install `sglang-omni` inside the container**

From a checkout of this repository:

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow -e .
```

## 🛠️ Option B: Manual install

Build prerequisites first:

- **UCX 1.20.x** with CUDA + verbs — [upstream](https://github.com/openucx/ucx), or reuse flags in [`docker/Dockerfile`](../../docker/Dockerfile).
- **flash-attn-4** `>=4.0.0b18`, matching `torch==2.13.0` and SGLang 0.5.20's `nvidia-cutlass-dsl` 4.6.2 pin.

Then:

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.6"
```

Latest on the index without a pin: `uv pip install --prerelease=allow sglang-omni`.

<a id="install-from-source"></a>

### Install from source

For Voicing-TTS, development, or unreleased changes:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow -v -e .   # drop -e for a non-editable install
```

## ✅ Verify with Voicing-TTS

Convert a Qwen3-TTS checkpoint into a Voicing-TTS checkpoint once, then serve
it:

```bash
python -m sglang_omni.models.voicing_tts.convert_checkpoint \
    Qwen/Qwen3-TTS-12Hz-1.7B-Base checkpoints/voicing-tts-12hz-1.7b-base

sgl-omni serve --config examples/configs/voicing_tts_1_7b.yaml --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "input": "Hello from Voicing-TTS.",
      "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "ref_text": "We asked over twenty different people, and they all said it was his."
    }' \
    --output output.wav
```

A Base checkpoint needs a reference clip; the checkpoint directory name must
contain `voicing-tts` and end in `base` for reference voices to be enabled.

See the [Voicing-TTS cookbook](../cookbook/voicing_tts.md) for the other
variants and request options.
