"""Backend-independent v2 animation orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from angels_ai.domain import AnimationRequest, AnimationResult, ConditioningBundle
from angels_ai.face import FacePreprocessor
from angels_ai.generation import AnimationBackend
from angels_ai.pose import PosePreprocessor


@dataclass
class AnimationPipeline:
    """Run preprocessing and generation for one animation request."""

    pose_preprocessor: PosePreprocessor
    backend: AnimationBackend
    face_preprocessor: Optional[FacePreprocessor] = None

    def run(self, request: AnimationRequest, work_dir: Path) -> AnimationResult:
        self._require_file(request.character_image, "character image")
        self._require_file(request.driver_video, "driver video")

        work_dir.mkdir(parents=True, exist_ok=True)
        request.output_video.parent.mkdir(parents=True, exist_ok=True)

        pose_video = self.pose_preprocessor.create_pose_video(
            request.driver_video,
            work_dir,
        )
        face_video = None
        if self.face_preprocessor is not None:
            face_video = self.face_preprocessor.create_face_video(
                request.driver_video,
                work_dir,
            )

        conditioning = ConditioningBundle(
            pose_video=pose_video,
            face_video=face_video,
        )
        video = self.backend.generate(request, conditioning)
        return AnimationResult(video=video, conditioning=conditioning)

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
