# Examples

Run these commands from the repository root after installing `sglang-omni`.

## Voicing-TTS Configs

`examples/configs/` holds one pipeline config per Voicing-TTS checkpoint. Each
config selects `VoicingTTSPipelineConfig` and points `model_path` at a converted
checkpoint directory under `checkpoints/`.

| Config | Converted checkpoint | Source checkpoint |
| --- | --- | --- |
| `voicing_tts_0_6b.yaml` | `checkpoints/voicing-tts-12hz-0.6b-base` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| `voicing_tts_1_7b.yaml` | `checkpoints/voicing-tts-12hz-1.7b-base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `voicing_tts_0_6b_customvoice.yaml` | `checkpoints/voicing-tts-12hz-0.6b-customvoice` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| `voicing_tts_1_7b_customvoice.yaml` | `checkpoints/voicing-tts-12hz-1.7b-customvoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `voicing_tts_1_7b_voicedesign.yaml` | `checkpoints/voicing-tts-12hz-1.7b-voicedesign` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |

The `_npu` variants (`voicing_tts_0_6b_npu.yaml`, `voicing_tts_1_7b_npu.yaml`,
`voicing_tts_0_6b_customvoice_npu.yaml`, `voicing_tts_1_7b_voicedesign_npu.yaml`)
use the same checkpoints with Ascend NPU settings.

## Convert a Checkpoint

Convert the source checkpoint (a local directory or Hugging Face repo id) into
the directory the config expects:

```bash
python -m sglang_omni.models.voicing_tts.convert_checkpoint \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  checkpoints/voicing-tts-12hz-1.7b-base
```

Keep `voicing-tts` in a Base checkpoint's directory name and end it with
`base`; the server enables uploaded reference voices from that path.

## Serve

```bash
sgl-omni serve --config examples/configs/voicing_tts_1_7b.yaml --port 8000
```

Pass `--model-path <converted dir>` to serve a checkpoint stored elsewhere. Use a
different `--port` if you run more than one server at the same time.

`examples/mps_dp/` launches several replicas on one GPU behind CUDA MPS; see
[MPS Data Parallelism](../docs/basic_usage/mps_dp.md).

See the [Voicing-TTS cookbook](../docs/cookbook/voicing_tts.md) for request
examples and tuning options.
