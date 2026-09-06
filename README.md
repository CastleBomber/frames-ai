# Dancing Angels AI

Turn a user-provided character image into walking, running, jumping, or dancing video.

## Golden Path: Apple Silicon

The first supported vertical slice is prompt-driven image-to-video generation:

`character image + action -> local MLX generation -> MP4`

After the local model is prepared:

```bash
source .venv-mlx/bin/activate
dancing-angels-ai doctor

dancing-angels-ai create \
  --image assets/man.png \
  --action walk \
  --output outputs/man-walking-mlx.mp4
```

Supported action presets: `walk`, `run`, `jump`, and `dance`. Use `--prompt`
when you want custom movement.

The default smoke-test settings are intentionally small for a 24 GB Apple-silicon
Mac: 512x512, 17 frames, 10 steps, and aggressive VAE tiling.
See [docs/mlx-runtime.md](docs/mlx-runtime.md) for one-time setup.

## Optional Exact-Motion Path

The existing Wan2.2-Animate integration remains available when exact motion from
a driver video matters:

`character image + driver video -> pose/face preprocessing -> Wan-Animate -> MP4`

That backend currently requires a configured NVIDIA CUDA worker:

```bash
dancing-angels-ai doctor --backend wan

dancing-angels-ai animate \
  --image assets/man.png \
  --motion assets/walk.mp4 \
  --output outputs/man-walking-wan.mp4
```

See [docs/wan-runtime.md](docs/wan-runtime.md) for its external runtime.

## Repository Layout

- `src/dancing_angels_ai/`: application code and generation backends.
- `experiments/pose/`: verified pose and motion prototypes.
- `experiments/legacy_sdxl/`: preserved SDXL and ControlNet experiments.
- `roadmaps-ideas/`: roadmaps, the Golden Image, and evolving product ideas.
- `assets/`: local source media.
- `models/`: local model files.
- `outputs/`: generated media.

## Tests

```bash
PYTHONPATH=src .venv-mlx/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```
