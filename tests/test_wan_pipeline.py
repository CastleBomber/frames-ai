"""Tests for the production Wan-Animate adapters and orchestration."""

import tempfile
import unittest
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from dancing_angels_ai.conditioning import (
    WanAnimatePreprocessor,
    WanPreprocessConfig,
    WanPreprocessError,
)
from dancing_angels_ai.domain import AnimationRequest, ConditioningBundle
from dancing_angels_ai.generation import WanAnimateBackend, WanGenerationConfig
from dancing_angels_ai.pipeline import AnimationPipeline
from dancing_angels_ai.video import inspect_video


def write_video(
    path: Path,
    size: Tuple[int, int],
    frame_count: int = 3,
    fps: float = 30.0,
) -> None:
    """Create a small readable MP4 fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("test MP4 codec is unavailable")
    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class WanAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "Wan2.2"
        self.checkpoint = self.root / "Wan2.2-Animate-14B"
        self.preprocess_script = (
            self.repository
            / "wan"
            / "modules"
            / "animate"
            / "preprocess"
            / "preprocess_data.py"
        )
        self.preprocess_script.parent.mkdir(parents=True)
        self.preprocess_script.touch()
        (self.repository / "generate.py").touch()
        (self.checkpoint / "process_checkpoint" / "det").mkdir(parents=True)
        pose_checkpoint = (
            self.checkpoint
            / "process_checkpoint"
            / "pose2d"
            / "vitpose_h_wholebody.onnx"
        )
        pose_checkpoint.mkdir(parents=True)
        (
            self.checkpoint
            / "process_checkpoint"
            / "det"
            / "yolov10m.onnx"
        ).touch()
        (pose_checkpoint / "end2end.onnx").touch()
        required_generation_files = [
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ]
        for relative_path in required_generation_files:
            (self.checkpoint / relative_path).touch()
        (self.checkpoint / "google" / "umt5-xxl").mkdir(parents=True)
        (self.checkpoint / "xlm-roberta-large").mkdir()
        for index in range(1, 5):
            (
                self.checkpoint
                / f"diffusion_pytorch_model-{index:05d}-of-00004.safetensors"
            ).touch()
        self.image = self.root / "man.png"
        self.image.write_bytes(b"fixture")
        self.driver = self.root / "walk.mp4"
        write_video(self.driver, (64, 96))
        self.output = self.root / "output.mp4"
        self.request = AnimationRequest(self.image, self.driver, self.output)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_official_preprocessor_command_and_artifacts(self) -> None:
        commands: List[Sequence[str]] = []

        def runner(command: Sequence[str], cwd: Path) -> None:
            commands.append(command)
            save_path = Path(command[command.index("--save_path") + 1])
            (save_path / "src_ref.png").write_bytes(b"fixture")
            write_video(save_path / "src_pose.mp4", (64, 96))
            write_video(save_path / "src_face.mp4", (512, 512))
            self.assertEqual(cwd, self.repository.resolve())

        preprocessor = WanAnimatePreprocessor(
            WanPreprocessConfig(
                wan_repository=self.repository,
                checkpoint_directory=self.checkpoint,
            ),
            runner=runner,
        )
        result = preprocessor.prepare(self.request, self.root / "conditioning")

        self.assertEqual(result.source_root, (self.root / "conditioning").resolve())
        self.assertEqual(result.face_video.name, "src_face.mp4")
        self.assertIn("--retarget_flag", commands[0])
        self.assertEqual(
            commands[0][commands[0].index("--fps") + 1],
            "-1",
        )

    def test_preprocessor_rejects_misaligned_face_video(self) -> None:
        def runner(command: Sequence[str], cwd: Path) -> None:
            save_path = Path(command[command.index("--save_path") + 1])
            (save_path / "src_ref.png").write_bytes(b"fixture")
            write_video(save_path / "src_pose.mp4", (64, 96), frame_count=3)
            write_video(save_path / "src_face.mp4", (512, 512), frame_count=2)

        preprocessor = WanAnimatePreprocessor(
            WanPreprocessConfig(self.repository, self.checkpoint),
            runner=runner,
        )
        with self.assertRaisesRegex(WanPreprocessError, "frame counts differ"):
            preprocessor.prepare(self.request, self.root / "conditioning")

    def test_generation_command_and_playable_output(self) -> None:
        source_root = self.root / "conditioning"
        source_root.mkdir()
        reference = source_root / "src_ref.png"
        reference.write_bytes(b"fixture")
        pose = source_root / "src_pose.mp4"
        face = source_root / "src_face.mp4"
        write_video(pose, (64, 96))
        write_video(face, (512, 512))
        commands: List[Sequence[str]] = []

        def runner(command: Sequence[str], cwd: Path) -> None:
            commands.append(command)
            save_file = Path(command[command.index("--save_file") + 1])
            write_video(save_file, (64, 96))

        backend = WanAnimateBackend(
            WanGenerationConfig(self.repository, self.checkpoint),
            runner=runner,
        )
        result = backend.generate(
            self.request,
            ConditioningBundle(
                pose_video=pose,
                face_video=face,
                reference_image=reference,
                source_root=source_root,
            ),
        )

        self.assertEqual(result, self.output.resolve())
        self.assertEqual(inspect_video(result).frame_count, 3)
        self.assertEqual(
            commands[0][commands[0].index("--task") + 1],
            "animate-14B",
        )


class AnimationPipelineTests(unittest.TestCase):
    def test_combined_conditioning_path_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            driver = root / "driver.mp4"
            output = root / "output.mp4"
            image.touch()
            driver.touch()
            pose = root / "pose.mp4"
            calls = []

            class Preprocessor:
                def prepare(self, request, work_dir):
                    calls.append(("prepare", request, work_dir))
                    return ConditioningBundle(pose_video=pose)

            class Backend:
                def generate(self, request, conditioning):
                    calls.append(("generate", request, conditioning))
                    output.touch()
                    return output

            pipeline = AnimationPipeline(
                pose_preprocessor=None,
                conditioning_preprocessor=Preprocessor(),
                backend=Backend(),
            )
            result = pipeline.run(
                AnimationRequest(image, driver, output),
                root / "work",
            )

            self.assertEqual(result.video, output)
            self.assertEqual([call[0] for call in calls], ["prepare", "generate"])

    def test_missing_character_fails_before_preprocessing(self) -> None:
        class UnusedBackend:
            def generate(self, request, conditioning):
                raise AssertionError("must not be called")

        pipeline = AnimationPipeline(
            pose_preprocessor=None,
            conditioning_preprocessor=None,
            backend=UnusedBackend(),
        )
        with self.assertRaisesRegex(FileNotFoundError, "character image"):
            pipeline.run(
                AnimationRequest(
                    Path("/missing/image.png"),
                    Path("/missing/video.mp4"),
                    Path("/missing/output.mp4"),
                ),
                Path("/tmp/unused"),
            )


if __name__ == "__main__":
    unittest.main()
