from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch
from PIL import Image

class SDEngine:
    """
    SDXL + ControlNet engine tuned for inference.
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

        # prefer float16 on accelerators (CUDA / MPS) to save memory
        torch_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        # load controlnet (SDXL-compatible weights)
        self.controlnet = ControlNetModel.from_pretrained(
            self.CONTROLNET_MODEL,
            torch_dtype=torch_dtype
        )

        # load SDXL ControlNet pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.SDXL_MODEL,
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=torch.float16
        )

        # move pipeline to device and enable inference optimizations
        self.pipe.to(self.device)

        # memory & speed tweaks
        try:
            # reduces memory usage at the cost of some speed
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        # enable xformers if available (very helpful on CUDA)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # on CUDA you can offload weights to CPU to reduce GPU memory
        if self.device == "cuda":
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                pass

    def generate_pose_frame(
            self, 
            text_prompt: str, 
            pose_image: Image.Image,
            negative_prompt: str = None,
            num_inference_steps: int = 40,
            guidance_scale: float = 5.0,
            seed: int = None,
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
            gen_device = self.device if self.device != "cpu" else "cpu"
            try:
                generator = torch.Generator(device=gen_device).manual_seed(seed)
            except Exception:
                # fallback for older torch on mps
                generator = torch.Generator().manual_seed(seed)

        # run pipeline
        out = self.pipe(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            image=pose_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # pipeline returns a Batch with .images
        img = out.images[0]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        return img

