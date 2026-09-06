"""RTMPose preprocessing for Dancing Angels AI driver videos."""

import logging
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PoseVideoError(RuntimeError):
    """Raised when pose conditioning cannot be created."""


class RTMPoseVideoPreprocessor:
    """Convert a driver video into frame-aligned pose conditioning."""

    def __init__(
        self,
        keypoint_threshold: float = 0.1,
        mode: str = "balanced",
        backend: str = "onnxruntime",
        codec: str = "mp4v",
        fallback_fps: float = 24.0,
        model: Optional[Any] = None,
    ) -> None:
        if not 0.0 <= keypoint_threshold <= 1.0:
            raise ValueError("keypoint_threshold must be between 0 and 1")
        if len(codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        if fallback_fps <= 0:
            raise ValueError("fallback_fps must be greater than zero")

        self.keypoint_threshold = keypoint_threshold
        self.mode = mode
        self.backend = backend
        self.codec = codec
        self.fallback_fps = fallback_fps
        self._model = model

    def create_pose_video(self, driver_video: Path, work_dir: Path) -> Path:
        """Create an MP4 pose video with the same timing and dimensions as input."""
        driver_video = Path(driver_video)
        work_dir = Path(work_dir)

        if not driver_video.is_file():
            raise FileNotFoundError(f"driver video not found: {driver_video}")

        work_dir.mkdir(parents=True, exist_ok=True)
        output_path = work_dir / f"{driver_video.stem}_pose.mp4"
        partial_path = work_dir / f".{driver_video.stem}_pose.partial.mp4"
        partial_path.unlink(missing_ok=True)

        capture = cv2.VideoCapture(str(driver_video))
        if not capture.isOpened():
            capture.release()
            raise PoseVideoError(f"could not open driver video: {driver_video}")

        writer = None
        total_frames = 0
        detected_frames = 0

        try:
            try:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise PoseVideoError(f"driver video has no readable frames: {driver_video}")

                height, width = frame.shape[:2]
                fps = self._read_fps(capture)
                writer = cv2.VideoWriter(
                    str(partial_path),
                    cv2.VideoWriter_fourcc(*self.codec),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise PoseVideoError(
                        f"could not create pose video with codec {self.codec}: {partial_path}"
                    )

                model = self._get_model()
                from rtmlib import draw_skeleton

                while ok and frame is not None:
                    keypoints, scores = model(frame)
                    pose_frame = np.zeros_like(frame)

                    if self._has_pose(keypoints):
                        pose_frame = draw_skeleton(
                            pose_frame,
                            np.asarray(keypoints),
                            np.asarray(scores),
                            kpt_thr=self.keypoint_threshold,
                        )
                        detected_frames += 1

                    writer.write(pose_frame)
                    total_frames += 1
                    ok, frame = capture.read()
            finally:
                capture.release()
                if writer is not None:
                    writer.release()

            if detected_frames == 0:
                raise PoseVideoError(f"no pose detected in driver video: {driver_video}")

            partial_path.replace(output_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Created pose video %s (%d/%d frames detected)",
            output_path,
            detected_frames,
            total_frames,
        )
        return output_path

    def _get_model(self) -> Any:
        if self._model is None:
            from rtmlib import Wholebody

            self._model = Wholebody(
                to_openpose=False,
                mode=self.mode,
                backend=self.backend,
            )
        return self._model

    def _read_fps(self, capture: cv2.VideoCapture) -> float:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            return self.fallback_fps
        return fps

    @staticmethod
    def _has_pose(keypoints: Any) -> bool:
        return keypoints is not None and np.asarray(keypoints).size > 0
