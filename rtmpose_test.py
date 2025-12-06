from rtmlib import RTMPose
from PIL imporpt Image
import numpy as np
import os

# Full path to the ONNX file
model_path = os.path.join("models", "rtmpose-s.onnx")

# Load model from local ONNX
pose_model = RTMPose(model_path)

# Load image
img_pil = Image.open("man.png")

# Convert to numpy array
img = np.array(img_pil)

# Run pose estimation, keypoints is a list of (x, y, confidence) for each detected joint
keypoints = pose_model(img)
print(keypoints)

# Visualize
pose_model.show_result(img, keypoints, save_path="output.png")

