"""Adapter for the official Wan2.2-Animate preprocessing pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from dancing_angels_ai.commands import CommandRunner, run_command
from dancing_angels_ai.domain import AnimationRequest, ConditioningBundle
from dancing_angels_ai.video import VideoMetadata, VideoValidationError, inspect_video


class WanPreprocessError(RuntimeError):
    """Raised when Wan-Animate conditioning cannot be prepared."""


@dataclass(frozen=True)
class WanPreprocessConfig:
    """Paths and options for official Wan-Animate preprocessing."""

    wan_repository: Path
    checkpoint_directory: Path
    python_executable: str = "python"
    resolution_width: int = 1280
    resolution_height: int = 720
    fps: int = -1
    retarget: bool = True
    use_flux: bool = False

    def __post_init__(self) -> None:
        if self.resolution_width <= 0 or self.resolution_height <= 0:
            raise ValueError("resolution dimensions must be greater than zero")
        if self.fps == 0 or self.fps < -1:
            raise ValueError("fps must be -1 (preserve source) or greater than zero")
        if self.use_flux and not self.retarget:
            raise ValueError("use_flux requires retarget=True")


class WanAnimatePreprocessor:
    """Produce Wan's exact src_ref/src_pose/src_face artifact contract."""

    def __init__(
        self,
        config: WanPreprocessConfig,
        runner: CommandRunner = run_command,
    ) -> None:
        self.config = config
        self.runner = runner

    def prepare(
        self,
        request: AnimationRequest,
        work_dir: Path,
    ) -> ConditioningBundle:
        """Run official preprocessing and validate its aligned outputs."""
        repository = self.config.wan_repository.expanduser().resolve()
        checkpoint = self.config.checkpoint_directory.expanduser().resolve()
        script = (
            repository
            / "wan"
            / "modules"
            / "animate"
            / "preprocess"
            / "preprocess_data.py"
        )
        process_checkpoint = checkpoint / "process_checkpoint"

        self._require_file(script, "Wan preprocessing script")
        self._require_file(
            process_checkpoint / "det" / "yolov10m.onnx",
            "Wan detection checkpoint",
        )
        self._require_pose_checkpoint(
            process_checkpoint / "pose2d" / "vitpose_h_wholebody.onnx"
        )
        if self.config.use_flux:
            self._require_directory(
                process_checkpoint / "FLUX.1-Kontext-dev",
                "Wan FLUX retarget checkpoint",
            )

        work_dir = Path(work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(request, work_dir, script, process_checkpoint)
        self.runner(command, repository)

        reference = work_dir / "src_ref.png"
        pose = work_dir / "src_pose.mp4"
        face = work_dir / "src_face.mp4"
        self._require_file(reference, "preprocessed reference image")

        try:
            pose_metadata = inspect_video(pose)
            face_metadata = inspect_video(face)
        except VideoValidationError as error:
            raise WanPreprocessError(str(error)) from error
        self._validate_alignment(pose_metadata, face_metadata)

        return ConditioningBundle(
            pose_video=pose,
            face_video=face,
            reference_image=reference,
            source_root=work_dir,
        )

    def build_command(
        self,
        request: AnimationRequest,
        work_dir: Path,
        script: Optional[Path] = None,
        process_checkpoint: Optional[Path] = None,
    ) -> Sequence[str]:
        """Build the official animation-mode preprocessing command."""
        repository = self.config.wan_repository.expanduser().resolve()
        script = script or (
            repository
            / "wan"
            / "modules"
            / "animate"
            / "preprocess"
            / "preprocess_data.py"
        )
        process_checkpoint = process_checkpoint or (
            self.config.checkpoint_directory.expanduser().resolve()
            / "process_checkpoint"
        )
        command = [
            self.config.python_executable,
            str(script),
            "--ckpt_path",
            str(process_checkpoint),
            "--video_path",
            str(request.driver_video.expanduser().resolve()),
            "--refer_path",
            str(request.character_image.expanduser().resolve()),
            "--save_path",
            str(work_dir),
            "--resolution_area",
            str(self.config.resolution_width),
            str(self.config.resolution_height),
            "--fps",
            str(self.config.fps),
        ]
        if self.config.retarget:
            command.append("--retarget_flag")
        if self.config.use_flux:
            command.append("--use_flux")
        return command

    @staticmethod
    def _validate_alignment(
        pose: VideoMetadata,
        face: VideoMetadata,
    ) -> None:
        if pose.frame_count != face.frame_count:
            raise WanPreprocessError(
                "Wan pose/face frame counts differ: "
                f"pose={pose.frame_count}, face={face.frame_count}"
            )
        if not abs(pose.fps - face.fps) < 0.01:
            raise WanPreprocessError(
                f"Wan pose/face FPS differ: pose={pose.fps}, face={face.fps}"
            )
        if (face.width, face.height) != (512, 512):
            raise WanPreprocessError(
                "Wan face conditioning must be 512x512; "
                f"got {face.width}x{face.height}"
            )

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise WanPreprocessError(f"{label} not found: {path}")

    @staticmethod
    def _require_directory(path: Path, label: str) -> None:
        if not path.is_dir():
            raise WanPreprocessError(f"{label} not found: {path}")

    @staticmethod
    def _require_pose_checkpoint(path: Path) -> None:
        """Accept Wan's single-file or external-data ONNX layout."""
        model_file = path / "end2end.onnx" if path.is_dir() else path
        if not model_file.is_file():
            raise WanPreprocessError(
                f"Wan whole-body pose checkpoint not found: {model_file}"
            )
