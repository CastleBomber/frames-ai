#!/usr/bin/env python3
"""
test_motion_video.py
--------------------

Step 6: Real motion extraction

What it does:
- Loads a video
- Runs RTMPose on every frame
- Creates pose-only conditioning images
- Saves a real motion sequence

Input:
- assets/walk.mp4

Output:
- outputs/poses/walk_pose_00.png
- outputs/poses/walk_pose_XX.png

Usage:
    python3 -m experiments.pose.test_motion_video assets/walk.mp4
"""

import os
import sys
import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton


def main(video_path: str, kpt_thr: float = 0.1):

    # ==============================================
    # VALIDATE INPUT
    # ==============================================
    if not os.path.exists(video_path):
        print("❌ Video not found")
        sys.exit(1)

    out_dir = "outputs/poses"
    os.makedirs(out_dir, exist_ok=True)

    # ==============================================
    # LOAD RTMPOSE
    # ==============================================
    print("🧠 Loading RTMPose...")
    model = Wholebody(
        to_openpose=False,
        mode="balanced",
        backend="onnxruntime"
    )

    # ==============================================
    # LOAD VIDEO
    # ==============================================
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Failed to open video")
        sys.exit(1)

    frame_idx = 0

    print("🎞️ Extracting motion...")

    # ==============================================
    # PROCESS VIDEO FRAMES
    # ==============================================
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        keypoints, scores = model(frame)

        if len(keypoints) == 0:
            continue

        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)

        black_bg = np.zeros_like(frame)

        pose = draw_skeleton(
            black_bg,
            keypoints,
            scores,
            kpt_thr=kpt_thr
        )

        out_path = os.path.join(
            out_dir,
            f"walk_pose_{frame_idx:02d}.png"
        )

        cv2.imwrite(out_path, pose)
        frame_idx += 1

    cap.release()

    print(f"✅ Wrote {frame_idx} pose frames")
    print(f"✅ Saved → {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python3 -m experiments.pose.test_motion_video walk.mp4")
        sys.exit(1)

    main(sys.argv[1])
