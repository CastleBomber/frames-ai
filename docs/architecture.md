# V2 Architecture

The v2 package follows the target pipeline without coupling core orchestration to Mixamo, Blender, RTMPose, or Wan-Animate.

## Package Boundaries

| Package | Responsibility |
|---|---|
| `dancing_angels_ai.motion` | Resolve an action such as walk or dance to a driver video. |
| `dancing_angels_ai.conditioning` | Run official Wan preprocessing for aligned reference, pose, and face artifacts. |
| `dancing_angels_ai.pose` | Preserve the verified RTMPose diagnostic/prototype path. |
| `dancing_angels_ai.face` | Preserve the generic face-preprocessor interface. |
| `dancing_angels_ai.generation` | Generate the character video from the image and conditioning. |
| `dancing_angels_ai.pipeline` | Validate inputs and orchestrate the complete flow. |

Mixamo and Blender are one motion-source implementation. Wan-Animate is one generation-backend implementation. These details stay behind small interfaces so either can be replaced later.

## Dependency Direction

`pipeline -> domain + conditioning + generation interfaces`

Experiment code must not be imported by `src/dancing_angels_ai/`. Useful experiment behavior should be refactored into v2 modules with focused tests before becoming production code.

## Implemented

- `WanAnimatePreprocessor` wraps Wan's official joint preprocessor. It consumes
  the character image and driver video once, then validates `src_ref.png`,
  `src_pose.mp4`, and the required 512x512 `src_face.mp4`.
- `WanAnimateBackend` invokes official `generate.py --task animate-14B` and
  validates the final MP4.
- `AnimationPipeline` supports the combined production preprocessor while
  retaining the earlier generic pose/face interfaces.
- `RTMPoseVideoPreprocessor` remains a useful diagnostic. Its output is not
  assumed to be interchangeable with Wan's official pose conditioning.

```bash
python3 -m dancing_angels_ai.pose assets/walk.mp4 --output-dir outputs/conditioning
```

The Wan checkout and its Python dependencies stay isolated from the application
environment. This avoids conflicts between Wan's pinned CUDA stack and the
legacy SDXL experiments.
