#!/usr/bin/env python3
"""
Test: RTMPose → Skeleton Render
Usage:
    python3 test-skeleton.py man.png
"""

import sys, os
import numpy as np
from PIL import Image, ImageDraw
from rtmlib import PoseModel 

# ---- CONFIG ----
TARGET_SIZE = (1024, 1024)

# COCO-style sample skeleton connections
# (you should later swap to the exact indexing your RTMPose model uses)
SKELETON_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # right arm
    (0,5),(5,6),(6,7),(7,8),          # left arm
    (0,9),(9,10),(10,11),(11,12),     # body/legs
]

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
    orig = Image.open(img_path).convert("RGB")
    w, h = orig.size

    # Convert to numpy for RTMPose
    np_img = np.array(orig)

    print("🧠 Running RTMPose...")
    model = PoseModel("rtmpose-s")  # or rtmpose-m
    result = model(np_img)

    keypoints = result["keypoints"][0]  # shape: (N, 3)

    print("🦴 Rendering skeleton...")
    canvas = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    sx = TARGET_SIZE[0] / w
    sy = TARGET_SIZE[1] / h

    # Draw joints
    for (x, y, conf) in keypoints:
        if conf < 0.1:
            continue
        xi, yi = int(x * sx), int(y * sy)
        r = 3
        draw.ellipse((xi-r, yi-r, xi+r, yi+r), fill=(255, 255, 255))

    # Draw bones
    for a, b in SKELETON_CONNECTIONS:
        if a < len(keypoints) and b < len(keypoints):
            xa, ya, ca = keypoints[a]
            xb, yb, cb = keypoints[b]
            if ca > 0.1 and cb > 0.1:
                draw.line(
                    (xa*sx, ya*sy, xb*sx, yb*sy),
                    fill=(255, 255, 255),
                    width=3
                )

    os.makedirs("tests", exist_ok=True)
    out_path = "tests/test-output.png"
    canvas.save(out_path)

    print(f"✅ Skeleton saved → {out_path}")


if __name__ == "__main__":
    main()


