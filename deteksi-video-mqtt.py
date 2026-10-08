import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt


# ============================================================
# PLATFORM
# ============================================================

SYSTEM = platform.system()          # "Windows", "Linux", "Darwin"
IS_WINDOWS = SYSTEM == "Windows"
IS_LINUX = SYSTEM == "Linux"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# ARGUMEN (opsional, default otomatis sesuai OS)
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="yolo11n_192.onnx")
parser.add_argument(
    "--camera",
    type=int,
    default=None,
    help="Index kamera. Default: 4 di Linux (STB), 0 di Windows",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Tampilkan jendela preview (default aktif di Windows)",
)
parser.add_argument(
    "--no-show",
    action="store_true",
    help="Matikan jendela preview",
)
args = parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = args.model
if not os.path.isabs(MODEL_PATH):
    # Cari model di folder script kalau tidak ada di folder kerja
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_PATH)

CAMERA_INDEX = (
    args.camera if args.camera is not None
    else (4 if IS_LINUX else 0)
)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

IMAGE_SIZE = 192
CONFIDENCE = 0.5
NMS_THRESHOLD = 0.45

SHOW_WINDOW = (IS_WINDOWS or args.show) and not args.no_show

STREAM_WIDTH = 320
STREAM_HEIGHT = 240
MQTT_FRAME_FPS = 5
JPEG_QUALITY = 50

MQTT_HOST = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_USERNAME = ""
MQTT_PASSWORD = ""
MQTT_TOPIC_FRAME = "yolo/frame"
MQTT_TOPIC_DETECTION = "yolo/detection"

NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


# ============================================================
# ONNX RUNTIME
# ============================================================

print(f"Platform: {SYSTEM} ({platform.machine()})", flush=True)
print("Loading ONNX Runtime...", flush=True)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model tidak ditemukan: {MODEL_PATH}", flush=True)
    sys.exit(1)

so = ort.SessionOptions()
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

if IS_LINUX and platform.machine().lower() in ("aarch64", "armv7l", "arm64"):
    # Konfigurasi stabil untuk ARM Cortex-A53 (STB) - JANGAN DIUBAH
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
else:
    # PC: pakai lebih banyak core
    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2)))
    so.inter_op_num_threads = 1

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

print(f"ONNX Runtime: {ort.__version__}", flush=True)
print(f"Input: {input_name} {input_shape}", flush=True)
print(
    f"Threads: intra={so.intra_op_num_threads}, inter={so.inter_op_num_threads}",
    flush=True,
)


# ============================================================
# MQTT (kompatibel paho-mqtt 1.x dan 2.x)
# ============================================================

client_id = "yolo-stb-" + str(int(time.time()))

try:
    mqtt_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
    )
except AttributeError:
    # paho-mqtt 1.x
    mqtt_client = mqtt.Client(client_id=client_id)

if MQTT_USERNAME:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)


def on_connect(client, userdata, flags, reason_code, *extra):
    print(f"MQTT connected: {reason_code}", flush=True)


def on_disconnect(client, userdata, *extra):
    print(f"MQTT disconnected: {extra}", flush=True)


mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

print(f"Connecting MQTT: {MQTT_HOST}:{MQTT_PORT}", flush=True)
mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
mqtt_client.loop_start()


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
    left = (IMAGE_SIZE - new_w) // 2
    top = (IMAGE_SIZE - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image, scale, left, top


# ============================================================
# POSTPROCESS
# ============================================================

def postprocess(output, original_width, original_height, scale, pad_x, pad_y):
    predictions = output

    if predictions.ndim == 3:
        predictions = predictions[0]

    # [84, N] -> [N, 84]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        if len(pred) < 6:
            continue

        cx, cy, bw, bh = pred[:4]
        class_scores = pred[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONFIDENCE:
            continue

        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        x1 = max(0, min(original_width - 1, x1))
        y1 = max(0, min(original_height - 1, y1))
        x2 = max(0, min(original_width - 1, x2))
        y2 = max(0, min(original_height - 1, y2))

        width = x2 - x1
        height = y2 - y1

        if width <= 1 or height <= 1:
            continue

        boxes.append([int(x1), int(y1), int(width), int(height)])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE, NMS_THRESHOLD)

    detections = []
    for index in np.array(indices).flatten():
        index = int(index)
        x, y, w, h = boxes[index]
        class_id = class_ids[index]
        name = NAMES[class_id] if 0 <= class_id < len(NAMES) else str(class_id)

        detections.append({
            "class_id": class_id,
            "name": name,
            "confidence": round(scores[index], 3),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    return detections


# ============================================================
# DRAW
# ============================================================

def draw_detections(frame, detections, fps):
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        label = f'{det["name"]} {det["confidence"]:.2f}'

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
        )

    cv2.putText(
        frame, f"YOLO FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Objects: {len(detections)}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )


# ============================================================
# CAMERA (lintas platform)
# ============================================================

def backends_for_platform():
    if IS_WINDOWS:
        return [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY),
        ]
    if IS_LINUX:
        return [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def try_open(index, backend, use_mjpg):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None

    # FOURCC harus di-set SEBELUM width/height
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Pastikan benar-benar bisa baca frame
    for _ in range(10):
        ok, _frame = cap.read()
        if ok:
            return cap
        time.sleep(0.05)

    cap.release()
    return None


def open_camera():
    # Index yang dicoba: yang diminta dulu, di Windows lanjut scan 0-3
    indices = [CAMERA_INDEX]
    if IS_WINDOWS:
        indices += [i for i in range(4) if i != CAMERA_INDEX]

    for index in indices:
        for name, backend in backends_for_platform():
            for use_mjpg in (True, False):
                cap = try_open(index, backend, use_mjpg)
                if cap is not None:
                    print(
                        f"Camera index {index} dibuka "
                        f"(backend={name}, MJPG={use_mjpg})",
                        flush=True,
                    )
                    return cap
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    cap = open_camera()

    if cap is None:
        print(
            "ERROR: Camera gagal dibuka. Coba --camera <index> yang lain "
            "dan pastikan kamera tidak dipakai aplikasi lain.",
            flush=True,
        )
        mqtt_client.loop_stop()
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {actual_width} x {actual_height}", flush=True)
    print(f"MQTT frame: {MQTT_TOPIC_FRAME}", flush=True)
    print(f"MQTT detection: {MQTT_TOPIC_DETECTION}", flush=True)
    print(
        f"Stream: {STREAM_WIDTH} x {STREAM_HEIGHT} @ {MQTT_FRAME_FPS} FPS",
        flush=True,
    )
    if SHOW_WINDOW:
        print("Preview aktif - tekan 'q' pada jendela untuk keluar", flush=True)

    previous_time = time.perf_counter()
    last_frame_publish = 0.0
    fail_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                fail_count += 1
                print("Camera read failed", flush=True)
                if fail_count > 100:
                    print("ERROR: kamera terus gagal, berhenti.", flush=True)
                    break
                time.sleep(0.05)
                continue
            fail_count = 0

            original_height, original_width = frame.shape[:2]

            input_tensor, scale, pad_x, pad_y = preprocess(frame)

            output = session.run(None, {input_name: input_tensor})

            detections = postprocess(
                output[0], original_width, original_height,
                scale, pad_x, pad_y,
            )

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-9)
            previous_time = current_time

            draw_detections(frame, detections, fps)

            detection_data = {
                "device": "yolo-stb",
                "timestamp": time.time(),
                "fps": round(fps, 2),
                "width": original_width,
                "height": original_height,
                "count": len(detections),
                "detections": detections,
            }

            mqtt_client.publish(
                MQTT_TOPIC_DETECTION,
                json.dumps(detection_data, separators=(",", ":")),
                qos=0,
                retain=False,
            )

            now = time.perf_counter()
            frame_size = 0

            if now - last_frame_publish >= 1.0 / MQTT_FRAME_FPS:
                stream_frame = cv2.resize(
                    frame, (STREAM_WIDTH, STREAM_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

                success, jpeg = cv2.imencode(
                    ".jpg", stream_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )

                if success:
                    jpeg_bytes = jpeg.tobytes()
                    mqtt_client.publish(
                        MQTT_TOPIC_FRAME, jpeg_bytes, qos=0, retain=False,
                    )
                    last_frame_publish = now
                    frame_size = len(jpeg_bytes)

            if SHOW_WINDOW:
                cv2.imshow("YOLO", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            object_summary = ", ".join(f'{d["name"]}:1' for d in detections)
            message = (
                f"FPS: {fps:5.1f} | "
                f"Objects ({len(detections)}): {object_summary or 'none'}"
            )
            if frame_size:
                message += f" | JPEG: {frame_size / 1024:.1f} KB"

            print(message, flush=True)

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)

    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("Camera released.", flush=True)
        print("MQTT disconnected.", flush=True)


if __name__ == "__main__":
    main()