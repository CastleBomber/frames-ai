"""Command-line entry point for pose-video preprocessing."""

import argparse
from pathlib import Path

from angels_ai.pose import RTMPoseVideoPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create pose conditioning from a driver video."
    )
    parser.add_argument("driver_video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/conditioning"))
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    preprocessor = RTMPoseVideoPreprocessor(keypoint_threshold=args.threshold)
    output_path = preprocessor.create_pose_video(args.driver_video, args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
