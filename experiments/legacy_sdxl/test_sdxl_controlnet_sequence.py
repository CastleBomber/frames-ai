#!/usr/bin/env python3
"""
test_sdxl_controlnet_sequence.py

Step 5: Multi-frame SDXL + ControlNet animation

What it does:
- Loads multiple pose conditioning images (OpenPose-style skeletons)
- Runs SDXL + ControlNet for each frame
- Keeps same seed for temporal consistency
- Generates a sequence of frames + an animated GIF

What it does:
- Loads multiple pose conditioning images (OpenPose-style skeletons)
- Runs SDXL + ControlNet for each frame
= Keeps same seed for temporal consistency
- Generates a sequence of frames + an aimated GIF

Input:
- tests/walk_pose_00.png ... tests/walk_pose_XX.png (pose frames)
- Text prompt describing character/style

Output:
- tests/sdxl_frame_00.png ... tests/sdxl_frame_XX.png
- tests/sdxl_walk.gif

Usage:
  cd /Users/cbombs/github/angels-ai
  source .venv/bin/activate

  python3 -m experiments.legacy_sdxl.test_sdxl_controlnet_sequence \
    --prompt "full body character, simple proportions, clean outline, flat colors, white background" \
    --negative "blurry, deformed, bad anatomy, messy background, inconsistent character" \
    --seed 123 \
    --steps 12 \
    --cfg 5.5 \
    --cond 1.3 \
    --size 512
"""

import os
import glob
import argparse
from PIL import Image
import imageio

from experiments.legacy_sdxl.app.diffusion.sd_engine import SDEngine


def main():

    # ==============================================
    # ARGUMENT PARSING
    # ============================================== 
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Text prompt describing the character/style")
    parser.add_argument("--negative", default="blurry, deformed, bad anatomy, extra limbs, extra fingers, messy background, clutter, inconsistent character")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--pose_glob", default="tests/walk_pose_*.png")
    parser.add_argument("--out_dir", default="tests")
    parser.add_argument("--gif_name", default="sdxl_walk.gif")
    parser.add_argument("--gif_ms", type=int, default=120)
    parser.add_argument("--size", type=int, default=768, help="Image size (width=height)")
    parser.add_argument("--cond", type=float, default=1.3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ==============================================
    # LOAD POSE FRAMES
    # ==============================================
    pose_paths = sorted(glob.glob(args.pose_glob))
    if not pose_paths:
        raise FileNotFoundError(f"No pose frames found matching: {args.pose_glob}")

    print(f"🧩 Found {len(pose_paths)} pose frames")

    # ==============================================
    # Initialize SD ENGINE
    # ==============================================
    sd = SDEngine()

    # ==============================================
    # GENERATE FRAMES
    # ==============================================
    out_frames = []
    for i, pose_path in enumerate(pose_paths):
        pose_img = Image.open(pose_path).convert("RGB")

        # SDXL typically likes 1024-ish sizes
        # To enforce 1024x1024:
        # pose_img = pose_img.resize((1024, 1024), Image.NEAREST)

        print(f"🎨 Generating frame {i:02d} from {os.path.basename(pose_path)} ...")
        frame = sd.generate_pose_frame(
            text_prompt=args.prompt,
            pose_image=pose_img.resize((args.size, args.size), Image.NEAREST),
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed,
            controlnet_conditioning_scale=args.cond,
            width=args.size,
            height=args.size,
        )

        out_path = os.path.join(args.out_dir, f"sdxl_frame_{i:02d}.png")
        frame.save(out_path)
        out_frames.append(frame)

    # ==============================================
    # BUILD GIF
    # ==============================================
    gif_path = os.path.join(args.out_dir, args.gif_name)

    imageio.mimsave(
        gif_path, 
            [f.convert("RGB") for f in out_frames], 
            duration=args.gif_ms / 1000.0
    )

    print(f"✅ Frames saved → {args.out_dir}/sdxl_frame_*.png")
    print(f"✅ GIF saved → {gif_path}")

if __name__ == "__main__":
    main()
