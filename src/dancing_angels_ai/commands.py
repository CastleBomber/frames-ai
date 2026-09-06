"""Safe subprocess helpers used by external model adapters."""

import shlex
import subprocess
from pathlib import Path
from typing import Protocol, Sequence


class ExternalCommandError(RuntimeError):
    """Raised when an external model command fails."""


class CommandRunner(Protocol):
    """Run an argument-vector command from a working directory."""

    def __call__(self, command: Sequence[str], cwd: Path) -> None:
        """Run the command or raise an exception."""
        ...


def run_command(command: Sequence[str], cwd: Path) -> None:
    """Run a command without a shell and surface a useful failure."""
    result = subprocess.run(list(command), cwd=str(cwd), check=False)
    if result.returncode != 0:
        rendered = " ".join(shlex.quote(part) for part in command)
        raise ExternalCommandError(
            f"external command exited with status {result.returncode}: {rendered}"
        )
