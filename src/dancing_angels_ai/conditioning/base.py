"""Combined conditioning interface for generation backends."""

from pathlib import Path
from typing import Protocol

from dancing_angels_ai.domain import AnimationRequest, ConditioningBundle


class ConditioningPreprocessor(Protocol):
    """Prepare every conditioning artifact needed by a backend."""

    def prepare(
        self,
        request: AnimationRequest,
        work_dir: Path,
    ) -> ConditioningBundle:
        """Return validated conditioning for one animation request."""
        ...
