# V2 Architecture

The v2 package follows the target pipeline without coupling core orchestration to Mixamo, Blender, RTMPose, or Wan-Animate.

## Package Boundaries

| Package | Responsibility |
|---|---|
| `angels_ai.motion` | Resolve an action such as walk or dance to a driver video. |
| `angels_ai.pose` | Convert the driver video into pose conditioning. |
| `angels_ai.face` | Convert the driver video into optional face conditioning. |
| `angels_ai.generation` | Generate the character video from the image and conditioning. |
| `angels_ai.pipeline` | Validate inputs and orchestrate the complete flow. |

Mixamo and Blender are one motion-source implementation. Wan-Animate is one generation-backend implementation. These details stay behind small interfaces so either can be replaced later.

## Dependency Direction

`pipeline -> domain + motion/pose/face/generation interfaces`

Experiment code must not be imported by `src/angels_ai/`. Useful experiment behavior should be refactored into v2 modules with focused tests before becoming production code.

## Implemented

`RTMPoseVideoPreprocessor` converts a driver MP4 into a pose-conditioning MP4. It preserves the source resolution, FPS, and frame count; frames without a detected person remain blank to preserve timing.

```bash
python3 -m angels_ai.pose assets/walk.mp4 --output-dir outputs/conditioning
```
