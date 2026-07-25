#!/usr/bin/env python3
"""
pose_engine.py
--------------

Core Engine: Command → Pose / Motion / Image Actions

What it does:
- Detects animation-related commands from user text
- Parses image filenames and optional size inputs
- Handles image transformations (pixelate)
- Generates simple pose sequences (walk / run / jump)
- Uses SDXL + ControlNet engine to turn pose frames into AI images
- Builds animated GIF outputs from generated frames

Pipeline Role:
- Bridges chat commands from main.py into actual visual actions
- Connects pose generation with diffusion rendering
- Serves as the core "animation assistant" controller

Inputs:
- User text commands
- Source images (.png / .jpg / .jpeg / .gif)
- Ex:
- pixelate man.png
- pixelate 64x64 man.png
- walk man.png
- run man.png
- jump man.png
- sit man.png
- turn man.png

Outputs:
- Pixelated images
- Pose frame sequences
- AI-generated animation GIFs

Usage:
- Used in main.py

    pose_engine = PoseEngine()

    user_input = "walk man.png"
    pose_engine.detect_pose(user_input)
    pose_engine.handle_pose("walk", user_input)

Notes:
- Current motion system uses simple offsets (prototype stage)
- Future upgrades:
    • Better biomechanics
    • Character consistency
    • Video export
    • Prompt memory


    
Example:

    user_input
"""

from PIL import Image           # opening images, resizing, pixelation
from rtmlib import RTMPose      # creates skeletons of characters
from experiments.legacy_sdxl.app.diffusion.sd_engine import SDEngine
import numpy as np
import re, os                   # parsing commands

class PoseEngine:
    """Detects pose or motion commands and handles them"""

    def __init__(self):
        self.rtmpose = RTMPose("models/rtmpose-s.onnx")
        self.sd = SDEngine()
        self.pose_keywords = {
            "pixelate": "Generate pixelated character",
            "walk": "Generate walking pose sequence",
            "run": "Generate running animation",
            "sit": "Generate seated pose",
            "jump": "Generate jumping pose",
            "turn": "Generate turning motion"
        }

    def detect_pose(self, text: str):
        """Return the matched pose keyword if found"""
        for keyword in self.pose_keywords:
            if re.search(rf"\b{keyword}\b", text.lower()):
                return keyword
        return None
    
    def generate_next_filename(self, base_filename):
        """
        Generate a new filename inside images/ 
        {base} + appending -1, -2, etc. to avoid overwriting
        """
        images_dir = "images"
        os.makedirs(images_dir, exist_ok=True)

        name, ext = os.path.splitext(os.path.basename(base_filename))
        n = 1
        while True:
            candidate = os.path.join(images_dir, f"{name}-{n}{ext}") # man-1.png             
            if not os.path.exists(candidate):
                return candidate
            n += 1

    def generate_gif_name(self, base_filename):
        """
        Ensures GIFs follow naming like images/man-1.gif
        """
        return self.generate_next_filename(os.path.join("images", f"{base_filename}.gif"))

    def resolve_image_path(self, filename):
        """
        Check if the requested image exists.

        Searches the current dir → then images/
        Return the resolved path or None if not found
        """
        # 1. Check current directory
        if os.path.exists(filename):
            return filename

        # 2. Check ./images/
        img_path = os.path.join("images", filename)
        if os.path.exists(img_path):
            return img_path

        return None 

    def pil_to_numpy(self, img):
        return np.array(img.convert("RGB"))

    def handle_pose(self, keyword, user_input=None):
        """Handle pose command based on keyword input"""
        action = self.pose_keywords[keyword]

        # --- Detect optional custom size, ex: "64x64"
        size_match = re.search(r"(\d+)\s*x\s*(\d+)", user_input or "")
        custom_size = (int(size_match.group(1)), int(size_match.group(2))) if size_match else None

        # --- Detect filename, ex: "man.png"
        match = re.search(r"([A-Za-z0-9_\-./]+\.(png|jpg|jpeg|gif))", user_input or "", re.IGNORECASE)
        image_name = match.group(1) if match else None # ex: "man.png"
        if not image_name:
            return f"❗ PoseEngine: No image filename detected in your command"
        
        # Resolve file path first (search current dir → then images/)
        resolved_path = self.resolve_image_path(image_name) # ex: "images/man.png" 
        if not resolved_path:
            return f"❌ PoseEngine: Image file not found — check the name or path: {image_name}"

        # --- Pixelate command
        if keyword == "pixelate":
            return self.pixelate_image(resolved_path, custom_size=custom_size)

        # --- Motion commands
        elif keyword in ["walk", "run", "jump"]:
            #images_dir = os.path.join("images", keyword)
            images_dir = "images"
            input_image = [ # (Array as a safety measure)
                os.path.join(images_dir, f) 
                for f in sorted(os.listdir(images_dir))
                if f.endswith(".png")
            ]

            if not input_image:
                # Fallback if no pose skeletons exist yet
                pose_frames = self.generate_pose_sequence(resolved_path)
                return self.create_sd_motion_gif(resolved_path, user_input, pose_frames)

            # Use Stable Diffusion animation
            return self.create_sd_motion_gif(resolved_path, user_input, input_image)

        return f"🩰 PoseEngine: {action} (simulation only for now)."

    def generate_skeleton_frame(self, img_path, frame_id):
        img = Image.open(img_path)

        np_img = self.pil_to_numpy(img)
        keypoints = self.rtmpose(np_img)

        draw = ImageDraw.Draw(img)

        # draw joints
        for (x, y, conf) in keypoints:
            draw.ellipse((x-3, y-3, x+3, y+3), fill="red")

        out_path = f"images/pose_{frame_id}.png"
        img.save(out_path)
        return out_path
    
    def generate_pose_sequence(self, img_path, num_frames=6):
        paths = []
        for i in range(num_frames):
            frame = self.generate_skeleton_frame(img_path, i)
            paths.append(frame)
        return paths

    def pixelate_image(self, image_path, custom_size=None):
        """
        Pixelates an image based on either:
        -A fixed custom size, ex: 64x64, OR
        -A pixelation factor (pixel_size)

        Prameters:
            resolved_path (str): Full filesystem path to the input image
            custom_size (tuple|None): Optional (width, height) override
            pixel_size (int): Pixelation strength when no custom size is given

        Example usage:
            pixelate [optional WxH] [filename]
            pixelate zeus.png
            pixelate 64x64 zeus.png
        """
        # --- Load the original image ---
        img = Image.open(image_path)

        # --- Choose pixel target dimensions ---
        target_size = custom_size if custom_size else (32, 32)

        # Downscale to small pixel grid
        small = img.resize(target_size, Image.NEAREST)

        # Upscale back to original size for blocky pixels
        pixelated = small.resize(img.size, Image.NEAREST)

        # Save result using next sequential filename
        base_filename = os.path.basename(image_path)
        pixel_image_name = self.generate_next_filename(base_filename)

        pixelated.save(pixel_image_name)

        return f"🩰 Pixelated image saved as {pixel_image_name}"

    def create_sd_motion_gif(self, image_path, user_prompt, pose_sequence_paths):
        """
        pose_sequence_paths: list of pose skeleton images (openpose PNGs)
        """
        frames = []

        for pose_img_path in pose_sequence_paths:
            pose_img = Image.open(pose_img_path)

            frame = self.sd.generate_pose_frame(
                text_prompt=user_prompt,
                pose_image=pose_img
            )

            frames.append(frame)

        # Builds the GIF a name, ex: images/man-1.gif
        base_filename = os.path.splitext(os.path.basename(image_path))[0]  # Set to "man"
        gif_name = self.generate_gif_name(base_filename)

        frames[0].save(
            gif_name,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0
        )

        return f"🧬 SD Motion GIF saved as {gif_name}"
