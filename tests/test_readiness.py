"""Tests for actionable Wan CUDA runtime readiness checks."""

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from dancing_angels_ai.readiness import (
    ProbeResult,
    check_mlx_runtime,
    check_wan_runtime,
)


class ReadinessTests(unittest.TestCase):
    def test_complete_mlx_runtime_reports_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "man.png"
            image.touch()
            model = root / "Wan2.2-TI2V-5B-MLX-Q4"
            model.mkdir()
            for filename in (
                "config.json",
                "model.safetensors",
                "t5_encoder.safetensors",
                "vae.safetensors",
            ):
                (model / filename).touch()

            def probe(command: Sequence[str], cwd: Path) -> ProbeResult:
                return ProbeResult(
                    0,
                    '{"python":"3.11.15","system":"Darwin",'
                    '"machine":"arm64","mlx":"0.32.0",'
                    '"mlx_video":"0.0.1"}\n',
                )

            report = check_mlx_runtime(
                model,
                "/usr/bin/python3",
                image,
                probe=probe,
            )

            self.assertTrue(report.ready, report.render())
            self.assertIn("[PASS] Apple Silicon", report.render())

    def test_complete_runtime_reports_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "Wan2.2"
            checkpoint = root / "Wan2.2-Animate-14B"
            image = root / "man.png"
            motion = root / "walk.mp4"
            image.touch()
            motion.touch()

            preprocess = (
                repository
                / "wan"
                / "modules"
                / "animate"
                / "preprocess"
                / "preprocess_data.py"
            )
            preprocess.parent.mkdir(parents=True)
            preprocess.touch()
            (repository / "generate.py").touch()
            (repository / "requirements_animate.txt").touch()

            required_files = [
                "config.json",
                "diffusion_pytorch_model.safetensors.index.json",
                "Wan2.1_VAE.pth",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "process_checkpoint/det/yolov10m.onnx",
                (
                    "process_checkpoint/pose2d/"
                    "vitpose_h_wholebody.onnx/end2end.onnx"
                ),
            ]
            for relative_path in required_files:
                path = checkpoint / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (checkpoint / "google" / "umt5-xxl").mkdir(parents=True)
            (checkpoint / "xlm-roberta-large").mkdir()
            for index in range(1, 5):
                (
                    checkpoint
                    / f"diffusion_pytorch_model-{index:05d}-of-00004.safetensors"
                ).touch()

            def probe(command: Sequence[str], cwd: Path) -> ProbeResult:
                if "-c" in command:
                    return ProbeResult(
                        0,
                        '{"python":"3.11.9","torch":"2.7.0","cuda":true,"devices":1}\n',
                    )
                return ProbeResult(0, "usage")

            report = check_wan_runtime(
                repository,
                checkpoint,
                "/usr/bin/python3",
                image,
                motion,
                probe=probe,
            )

            self.assertTrue(report.ready, report.render())
            self.assertIn("READY", report.render())

    def test_missing_configuration_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "man.png"
            motion = root / "walk.mp4"
            image.touch()
            motion.touch()

            def probe(command: Sequence[str], cwd: Path) -> ProbeResult:
                return ProbeResult(
                    0,
                    '{"python":"3.9.6","torch":"2.8.0","cuda":false,"devices":0}\n',
                )

            report = check_wan_runtime(
                None,
                None,
                "/usr/bin/python3",
                image,
                motion,
                probe=probe,
            )

            self.assertFalse(report.ready)
            rendered = report.render()
            self.assertIn("[FAIL] Wan repository: not configured", rendered)
            self.assertIn("[FAIL] NVIDIA CUDA", rendered)
            self.assertTrue(rendered.endswith("NOT READY"))


if __name__ == "__main__":
    unittest.main()
