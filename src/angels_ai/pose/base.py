"""Angels AI pose-preprocessing interface."""

from pathlib import Path
from typing import Protocol


class PosePreprocessor(Protocol):
    """Create pose conditioning from a driver video."""

    def create_pose_video(self, driver_video: Path, work_dir: Path) -> Path:
        """Return the generated pose-conditioning video."""
        ...
