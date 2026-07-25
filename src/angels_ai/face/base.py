"""Angels AI face-preprocessing interface."""

from pathlib import Path
from typing import Protocol


class FacePreprocessor(Protocol):
    """Create face conditioning from a driver video."""

    def create_face_video(self, driver_video: Path, work_dir: Path) -> Path:
        """Return the generated face-conditioning video."""
        ...
