# needs models/rtmpose_s.onnx rtmpose_m.onnx
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/rtmpose_s.onnx"

sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]   # M-series optimized
)

def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, 0)
    return img

def infer_keypoints(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    inp = preprocess(img)

    outputs = sess.run(None, {"input": inp})
    keypoints = outputs[0][0]  # shape: (17, 3)

    # Scale back up to original image size
    keypoints[:, 0] *= w
    keypoints[:, 1] *= h

    return keypoints


# Skeleton exactly like OpenPose ControlNet
SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
]

def render_skeleton(keypoints, out_path="skeleton.png"):
    canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # Draw joints
    for x, y, c in keypoints:
        if c > 0.3:
            cv2.circle(canvas, (int(x), int(y)), 5, (255, 255, 255), -1)

    # Draw bones
    for i, j in SKELETON:
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 > 0.3 and c2 > 0.3:
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                     (255, 255, 255), 3)

    cv2.imwrite(out_path, canvas)
    return out_path


# Full pipeline
def make_controlnet_pose(image_path):
    keypoints = infer_keypoints(image_path)
    return render_skeleton(keypoints, "pose_skeleton.png")


import cv2 
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/rtmpose_s.onnx"

sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"] # M-series optimized
)