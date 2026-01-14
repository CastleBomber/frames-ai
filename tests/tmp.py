#!/usr/bin/env python3
"""
This script takes a single character image and finds the main body joints (pose).
It then draws a stick-figure skeleton on top of the original image so you can confirm
pose detection is working before you generate animations.

Purpose:
- Load an input image
- Run RTMLib Wholebody (person detect + pose) using ONNX Runtime
- Overlay a skeleton on the original image for a quick visual check

Usage:
  source .venv/bin/activate
  python3 pose_sequence.py man.png

Output:
  tests/pose_vis.png
"""

import os
import sys
import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton

def main(img_path: str):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    os.makedirs("tests", exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"cv2.imread failed to load: {img_path}")

    # IMPORTANT:
    # Your current rtmlib draw_skeleton supports COCO-17 only.
    # So keep to_openpose=False to get a supported keypoint layout.
    wholebody = Wholebody(
        to_openpose=False,      # <- key fix (keeps COCO-17 output)
        mode="balanced",
        backend="onnxruntime"
    )

    keypoints, scores = wholebody(img)

    # Ensure numpy arrays with batch dimension
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    if keypoints.ndim == 2:      # (K, 2/3) -> (1, K, 2/3)
        keypoints = keypoints[None, ...]
    if scores.ndim == 1:         # (K,) -> (1, K)
        scores = scores[None, ...]

    vis = draw_skeleton(img.copy(), keypoints, scores, kpt_thr=0.1)

    out_path = os.path.join("tests", "pose_vis.png")
    cv2.imwrite(out_path, vis)
    print(f"✅ Wrote: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python3 pose_sequence.py <image.png>")
        sys.exit(1)
    main(sys.argv[1])
