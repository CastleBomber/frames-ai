# AGENTS.md

## Golden Goal
Turn a user-provided character image into testable walking, running, jumping,
or dancing videos.

## Decision Rule
The Golden Goal takes precedence over historical roadmaps, model choices, and
motion-source choices. Mixamo, Wan-Animate, and RTMPose are optional tools, not
product requirements.

## Current Increment
Prove a Mac-only vertical slice first:

`character image + action prompt -> MLX image-to-video -> playable MP4`

Then improve movement quality and add optional exact driver-video control.

## Working Rules
- Work incrementally and verify each vertical slice with a playable artifact.
- Preserve useful experiments and the existing CUDA Wan integration.
- Keep large models, user assets, generated outputs, and virtual environments ignored.
- Keep production code under `src/dancing_angels_ai/`.
- Do not treat roadmap completion as product completion.

## Roadmap Updates
- When updating a roadmap, move the superseded image into `roadmaps-ideas/archive/`.
- Add sequential letter suffixes within each roadmap version: `v5A`, `v5B`, `v5C`, and so on.
- Keep only the newest lettered image in `roadmaps-ideas/`; preserve older lettered images in the archive.
- Update related Markdown status and evidence files whenever the roadmap changes.
