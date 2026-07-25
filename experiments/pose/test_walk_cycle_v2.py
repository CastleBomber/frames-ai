#!/usr/bin/env python3
"""
test_walk_cycle_v2.py
---------------------

Step 3B: Improved procedural walk cycle

What it does:
- Loads a character image
- Detects pose keypoints once using RTMPose
- Generates smoother procedural walking motion
- Adds:
    - leg stride
    - knee bend
    - foot lift
    - torso bob
    - arm counter-swing
- Produces ControlNet-ready pose frames

Input:
- man.png

Output:
- tests/walk_v2_00.png ... walk_v2_11.png

Usage:
  cd /Users/cbombs/github/angels-ai
  source .venv/bin/activate

  python3 -m experiments.pose.test_walk_cycle_v2 man.png
"""

import os
import sys
import math
import cv2
import numpy as np

from rtmlib import Wholebody, draw_skeleton


# ==============================================
# COCO BODY INDICES
# ==============================================
L_SHOULDER = 5
R_SHOULDER = 6

L_ELBOW = 7
R_ELBOW = 8

L_WRIST = 9
R_WRIST = 10

L_HIP = 11
R_HIP = 12

L_KNEE = 13
R_KNEE = 14

L_ANKLE = 15
R_ANKLE = 16


def offset_joint(kp, idx, dx=0, dy=0):
    kp[idx, 0] += dx
    kp[idx, 1] += dy


def main(img_path: str, num_frames: int = 12, kpt_thr: float = 0.1):

    # ==============================================
    # VALIDATE INPUT
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
    # LOAD IMAGE
    # ==============================================
    img = cv2.imread(resolved)

    if img is None:
        print("❌ Failed to load image")
        sys.exit(1)

    h, w = img.shape[:2]

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
    # DETECT KEYPOINTS
    # ==============================================
    print("🦴 Detecting pose...")
    keypoints, scores = model(img)

    kp0 = np.asarray(keypoints)[0]
    sc0 = np.asarray(scores)[0]

    # ==============================================
    # WALK PARAMETERS
    # ==============================================
    stride_x = 0.045 * w
    foot_lift = 0.035 * h

    arm_swing = 0.03 * w
    torso_bob = 0.015 * h
    hip_sway = 0.01 * w

    # ==============================================
    # GENERATE WALK CYCLE
    # ==============================================
    print(f"🎞️ Generating {num_frames} frames...")

    for t in range(num_frames):

        phase = (2.0 * math.pi * t) / num_frames

        sin_p = math.sin(phase)
        cos_p = math.cos(phase)

        kp = kp0.copy()

        # ==========================================
        # TORSO MOTION
        # ==========================================
        kp[:, 1] += cos_p * torso_bob

        offset_joint(kp, L_HIP, -hip_sway * sin_p)
        offset_joint(kp, R_HIP, hip_sway * sin_p)

        # ==========================================
        # LEFT LEG
        # ==========================================
        offset_joint(kp, L_KNEE,
                     dx=-stride_x * sin_p,
                     dy=abs(cos_p) * foot_lift * 0.5)

        offset_joint(kp, L_ANKLE,
                     dx=-stride_x * sin_p,
                     dy=abs(cos_p) * foot_lift)

        # ==========================================
        # RIGHT LEG
        # ==========================================
        offset_joint(kp, R_KNEE,
                     dx=stride_x * sin_p,
                     dy=abs(-cos_p) * foot_lift * 0.5)

        offset_joint(kp, R_ANKLE,
                     dx=stride_x * sin_p,
                     dy=abs(-cos_p) * foot_lift)

        # ==========================================
        # ARM COUNTER-SWING
        # ==========================================
        offset_joint(kp, L_ELBOW, dx=arm_swing * sin_p)
        offset_joint(kp, L_WRIST, dx=arm_swing * sin_p)

        offset_joint(kp, R_ELBOW, dx=-arm_swing * sin_p)
        offset_joint(kp, R_WRIST, dx=-arm_swing * sin_p)

        # ==========================================
        # CLAMP TO IMAGE
        # ==========================================
        kp[:, 0] = np.clip(kp[:, 0], 0, w - 1)
        kp[:, 1] = np.clip(kp[:, 1], 0, h - 1)

        # ==========================================
        # DRAW SKELETON
        # ==========================================
        blank = np.zeros_like(img)

        vis = draw_skeleton(
            blank,
            kp[None, ...],
            sc0[None, ...],
            kpt_thr=kpt_thr
        )

        # ==============================================
        # OUTPUT: Save the generated images
        # ==============================================
        out_path = os.path.join("tests", f"walk_v2_{t:02d}.png")

        cv2.imwrite(out_path, vis)

    print("✅ Done.")
    print("✅ Wrote tests/walk_v2_*.png")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python3 -m experiments.pose.test_walk_cycle_v2 man.png")
        sys.exit(1)

    main(sys.argv[1])
