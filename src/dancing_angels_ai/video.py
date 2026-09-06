"""Small video-inspection helpers for artifact validation."""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoValidationError(RuntimeError):
    """Raised when a video artifact is missing, empty, or unreadable."""


@dataclass(frozen=True)
class VideoMetadata:
    """Properties needed to validate conditioning and final videos."""

    width: int
    height: int
    fps: float
    frame_count: int


def inspect_video(path: Path) -> VideoMetadata:
    """Return reliable basic metadata for a readable, nonempty video."""
    path = Path(path)
    if not path.is_file():
        raise VideoValidationError(f"video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(f"could not open video: {path}")

    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoValidationError(f"video has no readable frames: {path}")

        height, width = frame.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise VideoValidationError(f"video has invalid FPS: {path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = 1
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_count += 1
    finally:
        capture.release()

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )
