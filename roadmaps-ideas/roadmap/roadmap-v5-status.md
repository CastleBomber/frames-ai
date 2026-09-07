# Roadmap V5 Status

Goal: `Hero image + Shuffle -> local MLX -> playable MP4 in the website`

| Step | Goal | Status |
|---|---|---|
| 1 | Convert the MLX model | Complete — verified 2026-09-04 |
| 2 | Generate the first dance | Complete — playable MP4 verified 2026-09-05; visual limitations recorded |
| 3 | Set a quality baseline | Planned |
| 4 | Add local generation API | Planned |
| 5 | Connect the website | Planned |
| 6 | Add generation states | Planned |
| 7 | Play the real result | Planned |
| 8 | Package the Mac demo | Planned |

Acceptance gate: upload Hero, select Shuffle, generate, and watch a real MP4.

## Step 1 verification — 2026-09-04

- Original checkpoint preserved in `models/Wan2.2-TI2V-5B/`.
- Converted model: `models/Wan2.2-TI2V-5B-MLX-Q4/` (ignored by Git).
- Conversion exited 0; 300 transformer layers quantized to 4 bits, group size 64.
- All weight files passed tensor-header, shape, byte-count, and file-size checks.

| File | Tensors | Bytes | Precision |
|---|---:|---:|---|
| `model.safetensors` | 1,426 | 2,945,950,389 | Q4 targeted layers; remaining weights higher precision |
| `t5_encoder.safetensors` | 242 | 11,361,845,505 | BF16 |
| `vae.safetensors` | 196 | 2,818,778,952 | FP32 |

Conversion command:

```bash
.venv-mlx/bin/python -u -m mlx_video.models.wan_2.convert \
  --checkpoint-dir models/Wan2.2-TI2V-5B \
  --output-dir models/Wan2.2-TI2V-5B-MLX-Q4 \
  --quantize --bits 4 --group-size 64
```

The initial sandboxed attempt could not access Metal. Retrying with approved
GPU access completed successfully; no source-code changes were needed.

Readiness command: `.venv-mlx/bin/dancing-angels-ai doctor` — exit 0.

```text
[PASS] character image: assets/man.png
[PASS] MLX video model: /Users/cbombs/github/dancing-angels-ai/models/Wan2.2-TI2V-5B-MLX-Q4
[PASS] MLX model config.json
[PASS] MLX model model.safetensors
[PASS] MLX model t5_encoder.safetensors
[PASS] MLX model vae.safetensors
[PASS] MLX Python executable
[PASS] MLX Python version: 3.11.15
[PASS] Apple Silicon: Darwin arm64
[PASS] MLX imports: mlx=0.32.0, mlx-video=0.0.1
READY
```

Stopped before Step 2: no video generation or visual-quality validation performed.
Original planning poster:
`archive/roadmap-v5A-real-generation.png`. Previous update:
`archive/roadmap-v5B-real-generation-updated.png`. Current rebrand:
`roadmap-v5C-dancing-angels-real-generation-updated.png`.

## Step 2 verification — 2026-09-05

- Generated `outputs/v5-step2/20260905-000224/shuffle.mp4`: 512 × 512, 49 frames, 24 FPS, 2.041667 seconds.
- Seed 42; 40 steps; wall runtime 687.980 seconds (11m 28s).
- Full MP4 decode passed; sampled frames visually inspected.
- Hero clothing and overall identity mostly preserved; shuffle-like stepping visible. Hand/face distortion and motion-fidelity limitations remain.
- Failed-quality 10-step smoke clip preserved alongside the successful retry.
- [Inputs, settings, runtimes, playback checks, and visual findings](roadmap-v5-step2-evidence.md).
- Step 2 completes the first playable generation milestone, not a claim of production-quality or exact Shuffle reproduction. Stopped before Step 3.
