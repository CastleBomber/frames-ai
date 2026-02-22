#!/usr/bin/env python3
"""
Script: test_skeleton.py
Test: RTMPose → Skeleton Render
Version: 1.0

What it does:
- Loads an character image
- Runs RTMLib Wholebody (ONNXRuntime) to extract pose keypoints + confidence scores
- Identifies the body, limbs, and joints using an AI pose-detection model.
- Saves TWO outputs:
  1) tests/pose_overlay.png    (skeleton drawn on original image)
  2) tests/pose_condition.png  (skeleton drawn on a black background; ControlNet-ready)

Usage:
  source .venv/bin/activate
  python3 test_skeleton.py man.png

Notes:
- Uses OpenCV for image loading and saving.
- Uses ONNX Runtime for fast local inference.
- This script validates pose detection before feeding poses into animation pipelines
  (e.g., SDXL + ControlNet for motion generation).
"""


import sys, os
import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton 

# ---- MAIN TEST ----
def main(img_path: str):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        sys.exit(1)

    os.makedirs("tests", exist_ok=True)

    print(f"📸 Loading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image")
        sys.exit(1)

    # Wholebody model (auto downloads detector + pose if needed)
    print("🧠 Loading RTMPose (Wholebody) model...")
    wholebody = Wholebody(
        to_openpose=False,
        mode="balanced",            # good default
        backend="onnxruntime"       # CPU-safe
    )

    print("🦴 Running pose estimation...")
    keypoints, scores = wholebody(img)

    # Ensure numpy arrays (draw_skeleton expects arrays)
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    # --- Output 1: overlay on original image ---
    print("🎨 Rendering skeleton...")
    # Skeleton on original image (debug / sanity check)
    overlay = draw_skeleton(
        img.copy(), keypoints, scores, kpt_thr=0.1)

    overlay_path = os.path.join("tests", "pose-overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"✅ Wrote: {overlay_path}")

    # --- Output 2: pose-only conditioning image (black background) ---
    print("🧩 Rendering pose-only conditioning image (black background)...")
    black_bg = np.zeros_like(img)  # same HxWxC as input
    pose_condition = draw_skeleton(
        black_bg, keypoints, scores, kpt_thr=0.1)
    
    cond_path = os.path.join("tests", "pose_condition.png")
    cv2.imwrite(cond_path, pose_condition)
    print(f"✅ Wrote: {cond_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python3 test-skeleton.py <image.png>")
        sys.exit(1)
    main(sys.argv[1])

