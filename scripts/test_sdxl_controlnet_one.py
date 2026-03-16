#!/usr/bin/env python3
"""
test_sdxl_controlnet_one.py

Version: 1.3 (**Current script testing**)

Step 5A: One-frame SDXL+ControlNet sanity check (fastest debug).

What it does:
- Loads ONE pose frame (skeleton on black conditioning image)
- Prints pixel min/max so you know the pose isn’t literally blank
- Generates ONE SDXL+ControlNet output frame

Usage:
  cd /Users/cbombs/github/frames-ai
  source .venv/bin/activate

  python3 -m scripts.test_sdxl_controlnet_one \
  --pose tests/walk_pose_00.png \
  --prompt "character, clean outline, consistent design, plain background" \
  --seed 123 --steps 15 --size 512


Output:
  tests/sdxl_debug.png
"""

import os, argparse
import numpy as np
from PIL import Image
from app.diffusion.sd_engine import SDEngine

def main():
    # ==============================================
    # ARGUMENT PARSING
    # ==============================================
    p = argparse.ArgumentParser()
    p.add_argument("--pose", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="blurry, deformed, extra limbs, bad hands, low quality")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=4.5)
    p.add_argument("--cond", type=float, default=1.0)
    p.add_argument("--size", type=int, default=768)
    args = p.parse_args()

    os.makedirs("tests", exist_ok=True)

    # ==============================================
    # INPUT: Load and check pose image
    # ==============================================
    pose_img = Image.open(args.pose).convert("RGB").resize((args.size, args.size), Image.NEAREST)
    arr = np.asarray(pose_img)
    print(f"🧪 Pose pixels min={arr.min()} max={arr.max()} (max should be > 0)")

    # ==============================================
    # GENERATION: Run SDXL + ControlNet
    # ==============================================
    sd = SDEngine()
    out = sd.generate_pose_frame(
        text_prompt=args.prompt,
        pose_image=pose_img,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        controlnet_conditioning_scale=args.cond,
        width=args.size,
        height=args.size,
    )

    # ==============================================
    # OUTPUT: Save the generated image
    # ==============================================
    out_path = "tests/sdxl_debug.png"
    out.save(out_path)
    print(f"✅ Wrote: {out_path}")

if __name__ == "__main__":
    main()
