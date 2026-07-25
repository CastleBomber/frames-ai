#!/usr/bin/env python3
"""
test_openpose_condition.py
(unnecessary/ skip)
--------------------------

Step 4: True OpenPose conditioning

What it does:
- Loads a character image
- Detects body joints using RTMLib Wholebody
- Converts joints into an OpenPose-style skeleton map
- Creates a ControlNet-ready conditioning image

Input:
- man.png

Output:
- tests/openpose_condition.png

Usage:
    cd /Users/cbombs/github/angels-ai
    source .venv/bin/activate

    python3 -m scripts.test_openpose_condition man.png
"""

import os, sys
import cv2
import numpy as np
from rtmlib import Wholebody

# ==============================================
# OPENPOSE BODY CONNECTIONS (COCO-17 FORMAT)
# ==============================================
COCO_SKELETON = [x
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 6),              # shoulders
    (11, 12),            # hips
    (5, 11), (6, 12)     # torso
]

def main(img_path: str, kpt_thr: float = 0.1):

    # ==============================================
    # VALIDATE / RESOLVE INPUT FILE
    # ==============================================
    if os.path.exists(img_path):
        resolved = img_path
    elif os.path.exists(os.path.join("assets", img_path)):
        resolved = os.path.join("assets", img_path)
    else:
        print("❌ File not found")
        sys.exit(1)

    os.makedirs("tests", exist_ok=True)

    # ==============================================
    # LOAD SOURCE IMAGE
    # ==============================================
    img = cv2.imread(resolved)

    if img is None:
        print("❌ Failed to load image")
        sys.exit(1)

    h, w = img.shape[:2]

    # ==============================================
    # LOAD RTMPOSE MODEL
    # ==============================================
    print("🧠 Loading RTMPose (Wholebody)...")
    model = Wholebody(
        to_openpose=False,
        mode="balanced",
        backend="onnxruntime"
    )

    # ==============================================
    # DETECT BODY KEYPOINTS
    # ==============================================
    keypoints, scores = model(img)

    kp = np.asarray(keypoints)[0][:17]   # COCO-17 only
    sc = np.asarray(scores)[0][:17]

    # ==============================================
    # CREATE BLACK CANVAS
    # ==============================================
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # ==============================================
    # DRAW JOINT CIRCLES
    # ==============================================
    for i, (x, y) in enumerate(kp):
        if sc[i] >= kpt_thr:
            cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), -1)

    # ==============================================
    # DRAW BODY BONES / LIMBS
    # ============================================== 
    for a, b in COCO_SKELETON:
        if sc[a] >= kpt_thr and sc[b] >= kpt_thr:
            x1, y1 = kp[a]
            x2, y2 = kp[b]
            cv2.line(canvas,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     (255, 255, 255),
                     2)

    # ==============================================
    # OUTPUT: Save the generated image
    # ==============================================
    out_path = "tests/openpose_condition.png"
    cv2.imwrite(out_path, canvas)
    print(f"✅ Wrote: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m scripts.test_openpose_condition man.png")
        sys.exit(1)
    main(sys.argv[1])
