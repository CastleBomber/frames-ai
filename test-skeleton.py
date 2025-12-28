#!/usr/bin/env python3
"""
Test: RTMPose → Skeleton Render
Usage:
    source .venv/bin/activate
    python3 test-skeleton.py man.png
"""

import sys, os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from rtmlib import Wholebody, draw_skeleton, RTMPose 

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


    print("🧠 Loading RTMPose (Wholebody) model...")
    wholebody = Wholebody(
        to_openpose=False,
        mode="balanced",            # good default
        backend="onnxruntime"       # CPU-safe
    )

    print("🦴 Running pose estimation...")
    keypoints, scores = wholebody(img)

    print("🎨 Rendering skeleton...")
    vis = draw_skeleton(
        img.copy(),
        keypoints,
        scores,
        kpt_thr=0.1
    )

    os.makedirs("tests", exist_ok=True)
    out_path = "tests/pose_vis.png"
    cv2.imwrite(out_path, vis)

    print(f"✅ Skeleton saved to {out_path}")

if __name__ == "__main__":
    main()


