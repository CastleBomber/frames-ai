"""Animation-generation interface."""

from pathlib import Path
from typing import Protocol

from angels_ai.domain import AnimationRequest, ConditioningBundle


class AnimationBackend(Protocol):
    """Generate a character animation from prepared inputs."""

    def generate(
        self,
        request: AnimationRequest,
        conditioning: ConditioningBundle,
    ) -> Path:
        """Generate and return the final video path."""
        ...
