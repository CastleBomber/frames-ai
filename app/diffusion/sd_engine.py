#!/usr/bin/env python3
"""
********************************************************
    sd_engine.py

    SDXL + ControlNet engine for OpenPose conditioning
    Version: 2.0

    What it does:
    - Loads the AI model that turns a stick‑figure pose (like an OpenPose skeleton) into a full image
    - Automatically picks the best settings for your computer (GPU or CPU) to run fast and use less memory
    - Provides generate_pose_frame() to create a single image from a pose skeleton and prompt

*********************************************************
"""
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, EulerDiscreteScheduler
import torch
import numpy as np
from PIL import Image

# ==============================================
# SDENGINE CLASS
# ============================================== 
class SDEngine:
    """
    SDXL + ControlNet engine (OpenPose conditioning)
    """

    # Model choices
    SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_MODEL = "xinsir/controlnet-openpose-sdxl-1.0"

    def __init__(self, device: str = None):
        # --------------------------------------------
        # Device detection
        # -------------------------------------------- 
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Use float16 only on CUDA; float32 on MPS and CPU
        if self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32


        # --------------------------------------------
        # Load ControlNet and SDXL pipeline
        # --------------------------------------------
        self.controlnet = ControlNetModel.from_pretrained(
            self.CONTROLNET_MODEL,
            torch_dtype=self.dtype,
        )

        # Load SDXL ControlNet pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.SDXL_MODEL,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
        )

        # Move pipeline to device and enable inference optimizations
        self.pipe.to(self.device)

        # Set faster scheduler
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # --------------------------------------------
        # Memory & speed tweaks
        # --------------------------------------------
        try:
            # reduces memory usage at the cost of some speed
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        # Extra memory helper
        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass

        # CUDA-only speed/memory options
        if self.device == "cuda":
            # enable xformers if available (very helpful on CUDA)
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                pass

        # Warmup pass
        try:
            _ = self.pipe(
                prompt="warmup",
                image=Image.new("RGB", (1024, 1024), (0, 0, 0)),
                num_inference_steps=1,
            )
        except Exception:
            pass

    # ==============================================
    # GENERATE FRAME METHOD
    # ============================================== 
    def generate_pose_frame(
        self,
        text_prompt: str,
        pose_image: Image.Image,
        negative_prompt: str = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        seed: int = None,
        controlnet_conditioning_scale: float = 1.0,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        """
        Generate a single frame using SDXL + ControlNet openpose image.
        - text_prompt: textual description for the subject/scene
        - pose_image: PIL image used as the ControlNet conditioning (openpose skeleton)
        - returns: PIL.Image
        """

        # --------------------------------------------
        # Setup generator (for reproducible results)
        # -------------------------------------------- 
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except Exception:
                # fallback for older torch on mps
                generator = torch.Generator().manual_seed(seed)

        # --------------------------------------------
        # Prepare pose image
        # -------------------------------------------- 
        pose_image = pose_image.convert("RGB").resize((width, height), Image.NEAREST)

        # --------------------------------------------
        # Run pipeline
        # -------------------------------------------- 
        out = self.pipe(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            image=pose_image,  # control image for ControlNet
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        )

        # --------------------------------------------
        # Debug: print pixel stats
        # --------------------------------------------
        img = out.images[0]
        arr = np.array(img)
        print(f"🧪 Generated image: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")

        return out.images[0]
