"""Subprocess adapter for the official Wan2.2-Animate generator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from dancing_angels_ai.commands import CommandRunner, run_command
from dancing_angels_ai.domain import AnimationRequest, ConditioningBundle
from dancing_angels_ai.video import VideoValidationError, inspect_video


class WanGenerationError(RuntimeError):
    """Raised when Wan-Animate generation cannot produce a valid video."""


@dataclass(frozen=True)
class WanGenerationConfig:
    """Paths and inference options for the official Wan generator."""

    wan_repository: Path
    checkpoint_directory: Path
    python_executable: str = "python"
    previous_conditioning_frames: int = 1
    offload_model: bool = False
    convert_model_dtype: bool = False
    t5_cpu: bool = False

    def __post_init__(self) -> None:
        if self.previous_conditioning_frames not in (1, 5):
            raise ValueError("previous_conditioning_frames must be 1 or 5")


class WanAnimateBackend:
    """Generate an MP4 with the official Wan2.2 Animate-14B code."""

    def __init__(
        self,
        config: WanGenerationConfig,
        runner: CommandRunner = run_command,
    ) -> None:
        self.config = config
        self.runner = runner

    def generate(
        self,
        request: AnimationRequest,
        conditioning: ConditioningBundle,
    ) -> Path:
        """Run Wan animation mode and validate the resulting MP4."""
        repository = self.config.wan_repository.expanduser().resolve()
        checkpoint = self.config.checkpoint_directory.expanduser().resolve()
        script = repository / "generate.py"
        self._require_file(script, "Wan generator")
        self._validate_checkpoint(checkpoint)

        if conditioning.face_video is None:
            raise WanGenerationError("Wan face conditioning is required")
        if conditioning.reference_image is None:
            raise WanGenerationError("Wan processed reference image is required")
        if conditioning.source_root is None:
            raise WanGenerationError("Wan conditioning source directory is required")

        self._require_file(conditioning.pose_video, "Wan pose conditioning")
        self._require_file(conditioning.face_video, "Wan face conditioning")
        self._require_file(conditioning.reference_image, "Wan processed reference")

        output = request.output_video.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        partial_output = output.with_name(f".{output.stem}.partial{output.suffix}")
        partial_output.unlink(missing_ok=True)
        command = self.build_command(
            conditioning.source_root,
            partial_output,
            script,
        )
        try:
            self.runner(command, repository)
            inspect_video(partial_output)
            partial_output.replace(output)
        except VideoValidationError as error:
            partial_output.unlink(missing_ok=True)
            raise WanGenerationError(str(error)) from error
        except Exception:
            partial_output.unlink(missing_ok=True)
            raise
        return output

    def build_command(
        self,
        source_root: Path,
        output: Path,
        script: Optional[Path] = None,
    ) -> Sequence[str]:
        """Build the official single-GPU animation command."""
        repository = self.config.wan_repository.expanduser().resolve()
        script = script or repository / "generate.py"
        command = [
            self.config.python_executable,
            str(script),
            "--task",
            "animate-14B",
            "--ckpt_dir",
            str(self.config.checkpoint_directory.expanduser().resolve()),
            "--src_root_path",
            str(Path(source_root).expanduser().resolve()),
            "--refert_num",
            str(self.config.previous_conditioning_frames),
            "--save_file",
            str(Path(output).expanduser().resolve()),
        ]
        if self.config.offload_model:
            command.extend(["--offload_model", "True"])
        if self.config.convert_model_dtype:
            command.append("--convert_model_dtype")
        if self.config.t5_cpu:
            command.append("--t5_cpu")
        return command

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise WanGenerationError(f"{label} not found: {path}")

    @staticmethod
    def _require_directory(path: Path, label: str) -> None:
        if not path.is_dir():
            raise WanGenerationError(f"{label} not found: {path}")

    @classmethod
    def _validate_checkpoint(cls, checkpoint: Path) -> None:
        cls._require_directory(checkpoint, "Wan Animate checkpoint")
        required_files = [
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ]
        for relative_path in required_files:
            cls._require_file(
                checkpoint / relative_path,
                f"Wan checkpoint file {relative_path}",
            )
        for relative_path in ("google/umt5-xxl", "xlm-roberta-large"):
            cls._require_directory(
                checkpoint / relative_path,
                f"Wan checkpoint directory {relative_path}",
            )
        shards = list(checkpoint.glob("diffusion_pytorch_model-*.safetensors"))
        if len(shards) != 4:
            raise WanGenerationError(
                "Wan Animate checkpoint requires 4 transformer shards; "
                f"found {len(shards)} in {checkpoint}"
            )
