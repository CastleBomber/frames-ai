from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image

class SDEngine:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=torch.float16
        )

        self.pipe.to(device)
        self.device = device

    def generate_pose_frame(self, text_prompt, pose_image):
        """
        Generate a single SD frame based on text + pose skeleton
        """
        image = self.pipe(
            prompt=text_prompt,
            image=pose_image,   # openpose/pose skeleton
            num_inference_steps=25
        ).images[0]

        return image
    




from diffusers import StableDiffusionControlNetPipeline, ControlNewModel
import torch
from PIL import Image

class SDEngine:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=torch.float16
        )

        self.pipe.to(device)
        self.device = device

    def generate_pose_frame(self, text_prompt, pose_image):
        """
        Generate a single SD frame base on text + pose skeleton
        """
        image = self.pipe(
            prompt=text_prompt,
            image=pose_image, # openpose/pose skeleton
            num_inference_steps=25
        ).images[0]

        return image
