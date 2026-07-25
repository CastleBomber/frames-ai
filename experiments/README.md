# Experiments

These files are prototype evidence, not production modules.

## Pose And Motion

Verified during the Step 1 audit:

```bash
python3 -m experiments.pose.test_skeleton assets/man.png
python3 -m experiments.pose.test_walk_cycle_v2 assets/man.png
python3 -m experiments.pose.test_motion_video assets/walk.mp4
```

`test_motion_video.py` was refactored into `angels_ai.pose.RTMPoseVideoPreprocessor` during Roadmap V3 Step 4. The original remains here as evidence.

## Legacy SDXL

The old SDXL, ControlNet, and chat path is preserved under `legacy_sdxl/`. It is not the target backend and should not be extended.

## Archive

The archive contains incomplete, superseded, or empty scripts retained only for history.
