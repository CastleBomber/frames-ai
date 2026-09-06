"""Dancing Angels AI motion-source interface."""

from pathlib import Path
from typing import Protocol


class MotionSource(Protocol):
    """Resolve a named motion to a local driver video."""

    def get_driver_video(self, action: str, work_dir: Path) -> Path:
        """Return a driver video for the requested action."""
        ...
