#!/usr/bin/env python3
"""
Script: test-skeleton.py
Test: RTMPose → Skeleton Render
Version: 1.0

This script takes a single character image 
and identifies the body, limbs, and joints using an AI pose-detection model. 

It then draws the detected skeleton directly on top of the image 
so you can visually confirm that pose detection is working correctly.

Purpose:
- Verify RTMPose + ONNX Runtime + OpenCV are working correctly
- Visually confirm pose accuracy before using poses in animation pipelines
- Produce a clean skeleton overlay for debugging and inspection

Usage:
    source .venv/bin/activate
    python3 test-skeleton.py man.png

Output:
    tests/pose_vis.png  (original image with skeleton overlay)

Notes:
- Uses OpenCV for image loading and saving.
- Uses ONNX Runtime for fast local inference.
- This script validates pose detection before feeding poses into animation pipelines
  (e.g., SDXL + ControlNet for motion generation).
"""


import sys, os
import cv2
from rtmlib import Wholebody, draw_skeleton 

# ---- MAIN TEST ----
def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python3 test-skeleton.py <image.png>")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    print(f"📸 Loading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image")
        return

    # Wholebody model (auto downloads detector + pose if needed)
    print("🧠 Loading RTMPose (Wholebody) model...")
    wholebody = Wholebody(
        to_openpose=False,
        mode="balanced",            # good default
        backend="onnxruntime"       # CPU-safe
    )

    print("🦴 Running pose estimation...")
    keypoints, scores = wholebody(img)

    print("🎨 Rendering skeleton...")
    visualiation = draw_skeleton(img.copy(), keypoints, scores, kpt_thr=0.1)

    out_path = os.path.join("tests", "pose-skeleton-pose-visualiation.png")
    cv2.imwrite(out_path, visualiation)

    print(f"✅ Wrote: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python3 test-skeleton.py <image.png>")
        sys.exit(1)
    main(sys.argv[1])

