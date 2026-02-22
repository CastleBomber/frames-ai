import numpy as np
from PIL import Image
# ... other imports

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

    # --- DEBUG: Check generated image ---
    img = out.images[0]
    arr = np.array(img)
    print(f"🧪 Generated image: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
    return img