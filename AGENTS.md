# AGENTS.md

## Project Goal
Turn a user-provided character image into animated motion videos: run, dance, jump, walk.

## Current Direction
- Mixamo/Blender provides driver motion videos.
- Wan-Animate is the target generation backend.
- RTMPose / pose extraction work is prototype evidence.
- SDXL + ControlNet sequence code is legacy/prototype, not the preferred backend.

## Working Rules
- Audit before refactoring.
- Preserve useful prototype scripts.
- Do not delete roadmap, assets, or experiment history without approval.
- Prefer a clean v2 pipeline under `src/angels_ai/`.
- Keep old scripts under `legacy/` or `experiments/`.
- Keep assets and outputs ignored.

## Current Pipeline Target
`character image + Mixamo/Blender motion video -> pose/face preprocessing -> Wan-Animate -> MP4`

## Avoid
- Big rewrites before audit.
- Treating old scripts as production.
- Assuming Mixamo feeds Wan-Animate directly.
