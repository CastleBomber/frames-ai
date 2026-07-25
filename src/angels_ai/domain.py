"""Shared data contracts for the Angels AI v2 animation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AnimationRequest:
    """Inputs and destination for one animation run."""

    character_image: Path
    driver_video: Path
    output_video: Path
    action: Optional[str] = None


@dataclass(frozen=True)
class ConditioningBundle:
    """Driver-video signals consumed by a generation backend."""

    pose_video: Path
    face_video: Optional[Path] = None


@dataclass(frozen=True)
class AnimationResult:
    """Artifacts produced by one successful animation run."""

    video: Path
    conditioning: ConditioningBundle
