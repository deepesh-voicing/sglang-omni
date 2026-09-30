SGLang-Omni
=======================

SGLang-Omni is a high-performance serving framework for text-to-speech models, built on top of `SGLang <https://github.com/sgl-project/sglang>`_. It orchestrates multi-stage pipelines with low latency and OpenAI-compatible APIs. This build serves one model family: **Voicing-TTS**.

Speech-generation models decompose into heterogeneous stages with fundamentally different computational profiles: a CPU-bound text and reference-audio frontend, a memory-bound autoregressive talker, and a latency-sensitive vocoder. SGLang-Omni is built around a **computation-centric design**: each stage runs its own independent scheduler tuned to its bottleneck, communicates through a shared inbox/outbox abstraction, and transfers tensors via zero-copy shared memory. This prevents any single stage from degrading the others.

About
-----

Core features:

- **Multi-Stage Pipeline**: Flexible framework for orchestrating preprocessing, AR engine, and vocoder stages across processes and GPUs.
- **Native SGLang Integration**: Leverages SGLang's RadixAttention, continuous batching, and CUDA Graph optimizations for the AR backbone.
- **OpenAI-Compatible Server**: Drop-in ``/v1/audio/speech`` endpoint with streaming PCM output, plus batch synthesis, WebSocket streaming, and uploaded reference voices.
- **Voicing-TTS**: 12 Hz codec TTS in 0.6B and 1.7B sizes with Base (voice cloning), CustomVoice (preset speakers), and VoiceDesign (voice from a text description) variants.

Supported Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Model
     - Type
     - Notes
   * - `Voicing-TTS <cookbook/voicing_tts.html>`_ (converted from `Qwen/Qwen3-TTS-12Hz-1.7B-Base <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>`_ and sibling checkpoints)
     - TTS
     - Voice cloning, preset speakers, voice design, streaming, 0.6B / 1.7B


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation.md
   get_started/installation_npu.md
   get_started/installation_xpu.md
   get_started/installation_cpu.md
   get_started/installation_musa.md


.. toctree::
   :maxdepth: 1
   :caption: Cookbook

   cookbook/voicing_tts.md

.. toctree::
   :maxdepth: 1
   :caption: General Usage

   basic_usage/tts.md
   basic_usage/process_topology.md
   basic_usage/tts_process_topology.md
   basic_usage/process_topology_migration.md
   basic_usage/omni_router.md
   basic_usage/mps_dp.md


.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/relay.md
   benchmarks/voicing_tts_leading_silence.md


.. toctree::
   :maxdepth: 1
   :caption: Developer Reference

   developer_reference/main.md
   developer_reference/apiserver_design.md
   developer_reference/pipeline.md
   developer_reference/config.md
   developer_reference/adding_parameters.md
   developer_reference/communication.md
   developer_reference/reference_encode_service.md
   developer_reference/profiler.md
   developer_reference/rl_admin_control.md
   developer_reference/bump_version.md
