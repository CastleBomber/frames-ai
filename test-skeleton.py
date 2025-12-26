#!/usr/bin/env python3
"""
Test: RTMPose → Skeleton Render
Usage:
    source .venv/bin/activate
    python3 test-skeleton.py man.png
"""

import sys, os
import numpy as np
from PIL import Image, ImageDraw
from rtmlib import RTMPose 

# ---- CONFIG ----
TARGET_SIZE = (1024, 1024)

# COCO-style sample skeleton connections
# (Should later swap to the exact indexing your RTMPose model uses)
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

    print("🧠 Loading RTMPose model...")
    model = RTMPose(
        "models/rtmpose-s.onnx",
        model_input_size=(192,256), # (W, H)
    )

    # RTMLib handles resizing internally
    np_img = np.array(orig)
    outputs = model(np_img)

    # outputs shape: (num_people, num_keypoints, 4)
    keypoints = outputs[0]   # first person

    print("🦴 Rendering skeleton...")
    canvas = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    sx = TARGET_SIZE[0] / orig.width
    sy = TARGET_SIZE[1] / orig.height

    # Draw joints
    for kp in keypoints:
        x = float(kp[0])
        y = float(kp[1])
        conf = float(kp[2])

        if conf < 0.1:
            continue

        draw.ellipse(
            (x*sx-3, y*sy-3, x*sx+3, y*sy+3),
            fill=(255, 255, 255)
        )

    # Draw bones
    for a, b in SKELETON_CONNECTIONS:
        if a < len(keypoints) and b < len(keypoints):
            xa = float(keypoints[a][0][0])
            ya = float(keypoints[a][1][0])
            ca = float(keypoints[a][2][0])

            xb = float(keypoints[b][0][0])
            yb = float(keypoints[b][1][0])
            cb = float(keypoints[b][2][0])

            if ca > 0.1 and cb > 0.1:
                draw.line(
                    (xa*sx, ya*sy, xb*sx, yb*sy),
                    fill=(255, 255, 255),
                    width=3
                )

    os.makedirs("tests", exist_ok=True)
    canvas.save("tests/test-output.png")

    print("✅ Skeleton generated")

if __name__ == "__main__":
    main()


