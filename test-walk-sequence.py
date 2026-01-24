#!/usr/bin/env python3
"""
walk_sequence.py
Version: 1.1
Step 4: Generate a simple walk-cycle skeleton sequence from a single image.

What it does:
- Loads a single character image (e.g. man.png)
- Runs RTMLib Wholebody (ONNX Runtime) once to detect full-body pose keypoints
- Identifies the character’s body, limbs, and joints using an AI pose-detection model
- Generates a short walk-cycle by smoothly offsetting the detected keypoints over time
- Produces a sequence of skeleton-only images on a black background (ControlNet-ready)


Usage:
  source .venv/bin/activate
  python3 test-walk-sequence.py man.png

Creates:
  tests/walk_pose_00.png ... tests/walk_pose_11.png   (skeleton-on-black frames)

Notes:
- Uses RTMLib Wholebody (ONNX Runtime) to detect keypoints once.
- Then "animates" the pose by applying small, smooth x/y offsets to keypoints over time.
- This is meant to produce ControlNet-ready pose frames (not perfect biomechanics yet).
"""

import os, sys, math
import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton


def main(img_path: str, num_frames: int = 12, kpt_thr: float = 0.1):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        sys.exit(1)

    os.makedirs("tests", exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image with cv2.imread")
        sys.exit(1)

    h, w = img.shape[:2]

    print("🧠 Loading RTMPose (Wholebody) model...")
    wholebody = Wholebody(
        to_openpose=False,      
        mode="balanced",
        backend="onnxruntime"
    )

    print("🦴 Running pose estimation (single frame)...")
    keypoints, scores = wholebody(img)

    # Normalize to numpy + take first detected person
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    kp0 = keypoints[0] if keypoints.ndim == 3 else keypoints        # (K, 2)
    sc0 = scores[0] if scores.ndim == 2 else scores      # (K,)

    K = kp0.shape[0]
    print(f"ℹ️ Detected keypoints: {K} (Wholebody commonly returns 133)")

    # Compute a "centerline" so we can swing left vs right without hardcoded limb indices
    valid = sc0 >= kpt_thr

    # Motion strength (tweak freely)
    step_x = 0.06 * w   # leg swing strength
    arm_x  = 0.04 * w   # arm swing strength
    bob_y  = 0.015 * h  # vertical bob strength

    # In WholeBody-133, the FIRST 17 are the main COCO body joints.
    # COCO-17 indices:
    # 5=LShoulder 6=RShoulder 7=LElbow 8=RElbow 9=LWrist 10=RWrist
    # 11=LHip 12=RHip 13=LKnee 14=RKnee 15=LAnkle 16=RAnkle
    L_ARM = [5, 7, 9]
    R_ARM = [6, 8, 10]
    L_LEG = [11, 13, 15]
    R_LEG = [12, 14, 16]

    def swing(kp: np.ndarray, idx_list, dx, dy=0.0):
        for idx in idx_list:
            if idx < K and sc0[idx] >= kpt_thr:
                kp[idx, 0] += dx
                kp[idx, 1] += dy

    print(f"🎞️ Generating {num_frames} skeleton frames...")
    for t in range(num_frames):
        phase = math.sin(2.0 * math.pi * (t / num_frames))
        bob   = math.cos(2.0 * math.pi * (t / num_frames))

        kp = kp0.copy()

        # Legs swing opposite each other
        swing(kp, R_LEG, phase * step_x)     # right leg forward
        swing(kp, L_LEG, -phase * step_x)    # left leg backward

        # Arms swing opposite of legs (natural gait)
        swing(kp, R_ARM, -phase * arm_x)       # right arm back
        swing(kp, L_ARM, phase * arm_x)       # left arm forward

        # Whole-body bob
        kp[valid, 1] += bob * bob_y

        # Keep points in bounds (prevents drift off-canvas)
        kp[:, 0] = np.clip(kp[:, 0], 0, w - 1)
        kp[:, 1] = np.clip(kp[:, 1], 0, h - 1)

        # Draw skeleton on a BLACK background (ControlNet-friendly)
        blank = np.zeros_like(img)
        vis = draw_skeleton(
            blank,
            kp[None, ...],      # (1, K, 2)
            sc0[None, ...],     # (1, K)
            kpt_thr=kpt_thr
        )

        out_path = os.path.join("tests", f"walk_pose_{t:02d}.png")
        cv2.imwrite(out_path, vis)

    print("✅ Done. Skeleton sequence written to tests/walk_pose_*.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python3 test-walk-sequence.py <image.png>")
        sys.exit(1)
    main(sys.argv[1])
