def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python3 test-skeleton.py <image.png>")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    print(f"📸 Loading image: {img_path}")
    orig = Image.open(img_path).convert("RGB")

    print("🧠 Loading RTMPose model...")
    model = RTMPose(
        "models/rtmpose-s.onnx",
        model_input_size=(192, 256),  # (W, H)
    )

    np_img = np.array(orig)
    persons = model(np_img)
    keypoints = persons[0]  # shape: (K, 3) or (K, 4)

    print("🦴 Rendering skeleton...")
    canvas = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    sx = TARGET_SIZE[0] / orig.width
    sy = TARGET_SIZE[1] / orig.height

    # draw joints
    for kp in keypoints:
        x = float(kp[0])
        y = float(kp[1])
        conf = float(kp[2])   # ✅ FORCE SCALAR

        if conf < 0.1:
            continue

        draw.ellipse(
            (x*sx-3, y*sy-3, x*sx+3, y*sy+3),
            fill=(255, 255, 255)
        )

    # draw bones
    for a, b in SKELETON_CONNECTIONS:
        if a < len(keypoints) and b < len(keypoints):
            xa, ya, ca = map(float, keypoints[a][:3])
            xb, yb, cb = map(float, keypoints[b][:3])

            if ca > 0.1 and cb > 0.1:
                draw.line(
                    (xa*sx, ya*sy, xb*sx, yb*sy),
                    fill=(255, 255, 255),
                    width=3
                )

    os.makedirs("tests", exist_ok=True)
    canvas.save("tests/test-output.png")

    print("✅ Skeleton generated")
