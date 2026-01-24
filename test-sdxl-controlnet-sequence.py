#!/usr/bin/env python3
"""
test-sdxl-controlnet-sequence.py

Version: 1.3
Step 5: Turn pose frames into "AI animation" frames using SDXL + ControlNet(OpenPose).

What it does:
- Loads skeleton pose frames from tests/walk_pose_00.png ... tests/walk_pose_11.png
- Uses SDXL + ControlNet (OpenPose) via Diffusers to generate a matching image for each pose
- Saves:
  1) tests/sdxl_frame_00.png ... tests/sdxl_frame_11.png
  2) tests/sdxl_walk.gif

Usage:
  source .venv/bin/activate
  python3 test-sdxl-controlnet-sequence.py \
    --prompt "pixel art hero, clean outline, consistent character, plain background" \
    --seed 123 \
    --steps 30 \
    --cfg 5.0
"""

import os
import glob
import argparse
from PIL import Image
import imageio

from sd_engine import SDEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Text prompt describing the character/style")
    parser.add_argument("--negative", default="blurry, deformed, extra limbs, bad hands, low quality")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--pose_glob", default="tests/walk_pose_*.png")
    parser.add_argument("--out_dir", default="tests")
    parser.add_argument("--gif_name", default="sdxl_walk.gif")
    parser.add_argument("--gif_ms", type=int, default=120)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pose_paths = sorted(glob.glob(args.pose_glob))
    if not pose_paths:
        raise FileNotFoundError(f"No pose frames found matching: {args.pose_glob}")

    print(f"🧩 Found {len(pose_paths)} pose frames")

    sd = SDEngine()

    out_frames = []
    for i, pose_path in enumerate(pose_paths):
        pose_img = Image.open(pose_path).convert("RGB")

        # SDXL typically likes 1024-ish sizes
        # To enforce 1024x1024:
        # pose_img = pose_img.resize((1024, 1024), Image.NEAREST)

        print(f"🎨 Generating frame {i:02d} from {os.path.basename(pose_path)} ...")
        frame = sd.generate_pose_frame(
            text_prompt=args.prompt,
            pose_image=pose_img,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed,  # keep SAME seed across frames for consistency
        )

        out_path = os.path.join(args.out_dir, f"sdxl_frame_{i:02d}.png")
        frame.save(out_path)
        out_frames.append(frame)

    gif_path = os.path.join(args.out_dir, args.gif_name)
    imageio.mimsave(gif_path, [f.convert("RGB") for f in out_frames], duration=args.gif_ms / 1000.0)
    print(f"✅ Wrote frames: {args.out_dir}/sdxl_frame_*.png")
    print(f"✅ Wrote GIF: {gif_path}")


if __name__ == "__main__":
    main()
