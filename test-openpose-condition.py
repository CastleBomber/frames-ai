#!/usr/bin/env python3
"""
test-openpose-condition.py

Version: 1.2
Step 4: Generate OpenPose-style conditioning images for SDXL ControlNet.

What it does:
- Loads a character image
- Runs RTMLib Wholebody (ONNX Runtime) once
- Extracts COCO-17 body joints
- Renders an OpenPose-style skeleton on a black background
- Produces ConrtolNet-ready pose images

Usage:
    source .venv/bin/activate
    python3 test-openpose-condition.py man.png
"""

import os, sys
import cv2
import numpy as np
from rtmlib import Wholebody

# COCO-17 skeleton connections (OpenPose-style)
COCO_SKELETON = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6),              # shoulders
    (11, 12),            # hips
    (5, 11), (6, 12)     # torso
]

def main(img_path: str, kpt_thr: float = 0.1):
    if not os.path.exists(img_path):
        print("❌ File not found")
        sys.exit(1)

    os.makedirs("tests", exist_ok=True)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    print("🧠 Loading RTMPose (Wholebody)...")
    model = Wholebody(
        to_openpose=False,
        mode="balanced",
        backend="onnxruntime"
    )

    keypoints, scores = model(img)

    kp = np.asarray(keypoints)[0][:17]   # COCO-17 only
    sc = np.asarray(scores)[0][:17]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw joints
    for i, (x, y) in enumerate(kp):
        if sc[i] >= kpt_thr:
            cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), -1)

    # Draw bones
    for a, b in COCO_SKELETON:
        if sc[a] >= kpt_thr and sc[b] >= kpt_thr:
            x1, y1 = kp[a]
            x2, y2 = kp[b]
            cv2.line(canvas,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     (255, 255, 255),
                     2)

    out_path = "tests/openpose_condition.png"
    cv2.imwrite(out_path, canvas)
    print(f"✅ Wrote: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test-openpose-condition.py <image.png>")
        sys.exit(1)
    main(sys.argv[1])
