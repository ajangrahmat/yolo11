import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import Counter


MODEL = "yolo11n_192.onnx"
CAMERA_INDEX = 1
IMAGE_SIZE = 192
CONFIDENCE = 0.5


# ============================================================
# ONNX Runtime configuration
# ============================================================

so = ort.SessionOptions()

# 2 threads terbukti stabil dan paling cepat
so.intra_op_num_threads = 2
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    MODEL,
    sess_options=so,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

print("ONNX Runtime:", ort.__version__)
print("Input:", input_name)
print("Input shape:", session.get_inputs()[0].shape)
print("Threads: intra=2, inter=1")


# ============================================================
# COCO class names
# ============================================================

NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


# ============================================================
# YOLO preprocessing
# ============================================================

def preprocess(frame):
    h, w = frame.shape[:2]

    scale = min(
        IMAGE_SIZE / w,
        IMAGE_SIZE / h
    )

    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(
        frame,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        114,
        dtype=np.uint8,
    )

    x = (IMAGE_SIZE - nw) // 2
    y = (IMAGE_SIZE - nh) // 2

    canvas[y:y + nh, x:x + nw] = resized

    image = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB,
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    image = np.expand_dims(
        image,
        axis=0,
    )

    return image


# ============================================================
# YOLO postprocessing
# ============================================================

def postprocess(output):
    predictions = output[0]

    # YOLO11 output biasanya:
    # [1, 84, N]
    #
    # transpose menjadi:
    # [N, 84]

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:

        x, y, w, h = prediction[:4]

        class_scores = prediction[4:]

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )

        if confidence < CONFIDENCE:
            continue

        boxes.append(
            [
                float(x - w / 2),
                float(y - h / 2),
                float(w),
                float(h),
            ]
        )

        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONFIDENCE,
        0.45,
    )

    detections = []

    for index in indices:

        if isinstance(index, (list, tuple, np.ndarray)):
            index = int(index[0])
        else:
            index = int(index)

        detections.append(
            (
                class_ids[index],
                scores[index],
                boxes[index],
            )
        )

    return detections


# ============================================================
# Camera
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2,
)

if not cap.isOpened():
    raise RuntimeError("Camera gagal dibuka")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


print(
    "Camera:",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)


# ============================================================
# Main loop
# ============================================================

previous_time = time.perf_counter()

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("FAILED READ")
            break

        start = time.perf_counter()

        input_tensor = preprocess(frame)

        output = session.run(
            None,
            {
                input_name: input_tensor
            },
        )

        detections = postprocess(output)

        end = time.perf_counter()

        fps = 1.0 / max(
            end - previous_time,
            1e-9,
        )

        previous_time = end

        counts = Counter()

        for class_id, score, box in detections:

            if 0 <= class_id < len(NAMES):
                counts[NAMES[class_id]] += 1

        summary = ", ".join(
            f"{name}: {count}"
            for name, count in counts.items()
        )

        print(
            f"FPS: {fps:5.1f} | "
            f"Objects ({len(detections)}): "
            f"{summary or 'none'}",
            flush=True,
        )


except KeyboardInterrupt:

    print("\nStopped.")


finally:

    cap.release()