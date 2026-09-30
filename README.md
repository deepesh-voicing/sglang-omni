<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="logo" width="400"></img>

<p>
<a href="https://pypi.org/project/sglang-omni/"><img src="https://img.shields.io/pypi/v/sglang-omni?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
<a href="https://github.com/sgl-project/sglang-omni/stargazers"><img src="https://img.shields.io/github/stars/sgl-project/sglang-omni?style=for-the-badge&logo=github&label=stars" alt="GitHub stars"></a>
<a href="https://github.com/sgl-project/sglang-omni/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sgl-project/sglang-omni?style=for-the-badge" alt="license"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-closed-raw/sgl-project/sglang-omni?style=for-the-badge&label=closed%20issues" alt="closed issues"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-raw/sgl-project/sglang-omni?style=for-the-badge&label=open%20issues" alt="open issues"></a>
<a href="https://deepwiki.com/sgl-project/sglang-omni"><img src="https://img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge" alt="Ask DeepWiki"></a>
</p>

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://sgl-project.github.io/sglang-omni/"><b>Documentation</b></a> |
<a href="#quick-start"><b>Quick Start</b></a> |
<a href="https://sgl-project.github.io/sglang-omni/index.html"><b>Cookbook</b></a> |
<a href="https://github.com/sgl-project/sglang"><b>SGLang</b></a> |
<a href="https://slack.sglang.io"><b>Join Slack</b></a>
</p>

<p align="center">
⭐ <b><a href="https://github.com/sgl-project/sglang-omni/stargazers">Star SGLang-Omni</a> to help more builders discover open infrastructure for speech serving!</b>
</p>

## News

- [2026/09] 🔊 This build serves one model: **Voicing-TTS** (Base, CustomVoice, and VoiceDesign at 0.6B and 1.7B), converted from Qwen3-TTS 12Hz checkpoints. \[[Cookbook](./docs/cookbook/voicing_tts.md)\]
- [2026/09] 🚀 SGLang-Omni **v0.1.6** is on [PyPI](https://pypi.org/project/sglang-omni/). Install with `uv pip install --prerelease=allow "sglang-omni==0.1.6"`. \[[Installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)\]
- [2026/08] 🚀 TTS architecture refactor: shared pipeline state, engine construction, reference encoding, capability metadata, and vocoder scheduling. \[[Roadmap](https://github.com/sgl-project/sglang-omni/issues/985)\] \[[Blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-refactor.md)\]

## About

SGLang-Omni is a multi-stage serving runtime for TTS models. Its design target is multi-stage decoding: generation split across heterogeneous stages with different compute patterns, dependency structures, and resource needs. SGLang-Omni owns the pipeline topology, stage lifecycle, inter-stage transport, model integration layer, and OpenAI-compatible serving surface, while composing with [SGLang](https://github.com/sgl-project/sglang) for high-performance autoregressive scheduling and model execution.

- **Multi-stage runtime**: SGLang-Omni models generation as coordinated stages: preprocessing, an autoregressive TTS engine, and a streaming vocoder.
- **Stage-specialized scheduling**: Each stage runs behind a scheduler matched to its workload, from SGLang-backed autoregressive scheduling to lightweight preprocessing and streaming vocoder loops.
- **Transport-aware execution**: A control plane coordinates requests while the relay data plane moves tensor payloads across shared-memory, NCCL, NIXL, and Mooncake backends.
- **API surface**: OpenAI-compatible endpoints expose speech generation, batch speech, streaming speech (HTTP and WebSocket), and uploaded voices.

## What SGLang-Omni Serves

- **Speech generation**: [Voicing-TTS](./docs/cookbook/voicing_tts.md) — voice cloning (Base), built-in speakers (CustomVoice), and voice design (VoiceDesign) on `/v1/audio/speech`, batch, streaming, uploaded voices.
- **SGLang-Omni Router**: Multi-worker OpenAI-compatible front door — health, readiness, lifecycle, capability discovery. [Router guide](./docs/basic_usage/omni_router.md).

## Hardware Support

| Backend | Status | Notes |
|---------|--------|-------|
| **NVIDIA CUDA** | Supported | Default backend. |
| **Ascend NPU** | Experimental | Voicing-TTS through the `examples/configs/voicing_tts_*_npu.yaml` configs. Install per [Ascend NPU guide](./docs/get_started/installation_npu.md). |
| **Intel GPU (XPU)** | Experimental | Intel Arc GPUs via PyTorch XPU. Voicing-TTS serves end-to-end. Install per [Intel XPU guide](./docs/get_started/installation_xpu.md); the backend is auto-detected. |
| **Apple Silicon** | Not supported | The macOS path served ASR only, which this build no longer includes. |

## Quick Start

Install SGLang-Omni per [Installation](./docs/get_started/installation.md), convert a checkpoint, and serve it:

```bash
python -m sglang_omni.models.voicing_tts.convert_checkpoint \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  checkpoints/voicing-tts-12hz-1.7b-base

sgl-omni serve --config examples/configs/voicing_tts_1_7b.yaml --port 8000
```

Then clone a voice from a reference clip:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voicing-tts",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

- [Voicing-TTS cookbook](./docs/cookbook/voicing_tts.md)
- [TTS usage](./docs/basic_usage/tts.md)
- [Omni router](./docs/basic_usage/omni_router.md)
- [Developer reference](./docs/developer_reference/main.md)

## Community & Support

SGLang-Omni welcomes contributors working on inference systems, kernels, scheduling, inter-stage communication, model runners and cache efficiency, model integration, benchmarking, production deployment. Join the [SGLang Slack](https://slack.sglang.io) or read the [developer reference](https://sgl-project.github.io/sglang-omni/developer_reference/main.html).

Organizations interested in supporting SGLang-Omni or TTS serving can contact Chenyang Zhao at [zhaochenyang@lmsys.org](mailto:zhaochenyang@lmsys.org).

## Acknowledgments

SGLang-Omni builds on the SGLang ecosystem and on open model work from the TTS and speech communities. Voicing-TTS checkpoints are converted from the Qwen3-TTS checkpoints released by the Qwen team. We thank the model teams, systems contributors, and partner organizations helping make open speech serving faster, more reliable, and easier to extend.
