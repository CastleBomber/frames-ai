#!/usr/bin/env python3
"""
********************************************************
    sd_engine.py
    Version: 1.3 (current testing environment)
    Version: 2.0 (compatible)
*********************************************************
"""
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch
from PIL import Image


class SDEngine:
    """
    SDXL + ControlNet inference engine.
    - Uses SDXL base model + an OpenPose SDXL controlnet.
    - Exposes generate_pose_frame(...) for single-frame generation.
    """

    # Model choices
    SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_MODEL = "xinsir/controlnet-openpose-sdxl-1.0"

    def __init__(self, device: str = None):
        # pick run-time device (auto-detect)
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Float16 only on cuda/mps; cpu should be float32
        self.dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        # Load controlnet (SDXL-compatible weights)
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

        # Memory & speed tweaks
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

        # (Optional) Warmup pass
        try:
            _ = self.pipe(
                prompt="warmup",
                image=Image.new("RGB", (1024, 1024), (0, 0, 0)),
                num_inference_steps=1
            )
        except Exception:
            pass

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
        

        # deterministic generator when seed provided
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except Exception:
                # fallback for older torch on mps
                generator = torch.Generator().manual_seed(seed)

        pose_image = pose_image.convert("RGB").resize((width, height), Image.NEAREST)

        # run pipeline
        out = self.pipe(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            image=pose_image,  # control image for ControlNet
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        )

        return out.images[0]
