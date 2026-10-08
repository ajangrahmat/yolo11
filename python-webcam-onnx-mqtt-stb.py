import time
import json

import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "yolo11n_192.onnx"

CAMERA_INDEX = 4
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

IMAGE_SIZE = 192

CONFIDENCE = 0.5
NMS_THRESHOLD = 0.45


# ============================================================
# STREAM CONFIG
# ============================================================

# Resolusi khusus untuk frame MQTT.
# TIDAK mengubah resolusi inference YOLO.
STREAM_WIDTH = 320
STREAM_HEIGHT = 240

MQTT_FRAME_FPS = 5

# 50 = ukuran JPEG lebih kecil
JPEG_QUALITY = 50


# ============================================================
# MQTT
# ============================================================

MQTT_HOST = "broker.emqx.io"
MQTT_PORT = 1883

MQTT_USERNAME = ""
MQTT_PASSWORD = ""

MQTT_TOPIC_FRAME = "yolo/frame"
MQTT_TOPIC_DETECTION = "yolo/detection"


# ============================================================
# COCO CLASS
# ============================================================

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
    "hair drier", "toothbrush"
]


# ============================================================
# ONNX RUNTIME
# ============================================================

print("Loading ONNX Runtime...", flush=True)

so = ort.SessionOptions()

# ============================================================
# JANGAN DIUBAH
# Ini konfigurasi yang sudah terbukti stabil
# di ARM Cortex-A53 kamu.
# ============================================================

so.intra_op_num_threads = 2
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL


session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CPUExecutionProvider"],
)


input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape


print(
    f"ONNX Runtime: {ort.__version__}",
    flush=True,
)

print(
    f"Input: {input_name}",
    flush=True,
)

print(
    f"Input shape: {input_shape}",
    flush=True,
)

print(
    "Threads: intra=2, inter=1",
    flush=True,
)


# ============================================================
# MQTT
# ============================================================

mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=(
        "yolo-stb-"
        + str(int(time.time()))
    ),
)


if MQTT_USERNAME:

    mqtt_client.username_pw_set(
        MQTT_USERNAME,
        MQTT_PASSWORD,
    )


def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties,
):

    print(
        f"MQTT connected: {reason_code}",
        flush=True,
    )


def on_disconnect(
    client,
    userdata,
    disconnect_flags,
    reason_code,
    properties,
):

    print(
        f"MQTT disconnected: {reason_code}",
        flush=True,
    )


mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect


print(
    f"Connecting MQTT: "
    f"{MQTT_HOST}:{MQTT_PORT}",
    flush=True,
)


mqtt_client.connect(
    MQTT_HOST,
    MQTT_PORT,
    keepalive=60,
)


mqtt_client.loop_start()


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(frame):

    h, w = frame.shape[:2]

    scale = min(
        IMAGE_SIZE / w,
        IMAGE_SIZE / h,
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        114,
        dtype=np.uint8,
    )

    left = (
        IMAGE_SIZE - new_w
    ) // 2

    top = (
        IMAGE_SIZE - new_h
    ) // 2

    canvas[
        top:top + new_h,
        left:left + new_w
    ] = resized

    # BGR -> RGB
    image = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB,
    )

    # uint8 -> float32
    image = (
        image.astype(
            np.float32
        ) / 255.0
    )

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1),
    )

    # Add batch
    image = np.expand_dims(
        image,
        axis=0,
    )

    return (
        image,
        scale,
        left,
        top,
    )


# ============================================================
# POSTPROCESS
# ============================================================

def postprocess(
    output,
    original_width,
    original_height,
    scale,
    pad_x,
    pad_y,
):

    predictions = output

    if predictions.ndim == 3:

        predictions = predictions[0]

    # YOLO biasanya:
    # [84, 8400]
    #
    # menjadi:
    # [8400, 84]

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


        class_id = int(
            np.argmax(
                class_scores
            )
        )


        confidence = float(
            class_scores[class_id]
        )


        if confidence < CONFIDENCE:
            continue


        # xywh -> xyxy

        x1 = cx - bw / 2
        y1 = cy - bh / 2

        x2 = cx + bw / 2
        y2 = cy + bh / 2


        # Remove letterbox padding

        x1 = (
            x1 - pad_x
        ) / scale

        y1 = (
            y1 - pad_y
        ) / scale

        x2 = (
            x2 - pad_x
        ) / scale

        y2 = (
            y2 - pad_y
        ) / scale


        # Clip

        x1 = max(
            0,
            min(
                original_width - 1,
                x1,
            ),
        )

        y1 = max(
            0,
            min(
                original_height - 1,
                y1,
            ),
        )

        x2 = max(
            0,
            min(
                original_width - 1,
                x2,
            ),
        )

        y2 = max(
            0,
            min(
                original_height - 1,
                y2,
            ),
        )


        width = x2 - x1
        height = y2 - y1


        if width <= 1 or height <= 1:
            continue


        boxes.append(
            [
                int(x1),
                int(y1),
                int(width),
                int(height),
            ]
        )

        scores.append(
            confidence
        )

        class_ids.append(
            class_id
        )


    if not boxes:
        return []


    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONFIDENCE,
        NMS_THRESHOLD,
    )


    detections = []


    for index in indices:

        if isinstance(
            index,
            (
                list,
                tuple,
                np.ndarray,
            ),
        ):

            index = int(
                index[0]
            )

        else:

            index = int(index)


        x, y, w, h = boxes[index]

        class_id = class_ids[index]

        name = (
            NAMES[class_id]
            if 0 <= class_id < len(NAMES)
            else str(class_id)
        )


        detections.append(
            {
                "class_id": class_id,
                "name": name,
                "confidence": round(
                    scores[index],
                    3,
                ),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )


    return detections


# ============================================================
# DRAW
# ============================================================

def draw_detections(
    frame,
    detections,
    fps,
):

    for det in detections:

        x = det["x"]
        y = det["y"]
        w = det["w"]
        h = det["h"]


        label = (
            f'{det["name"]} '
            f'{det["confidence"]:.2f}'
        )


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )


        cv2.putText(
            frame,
            label,
            (
                x,
                max(
                    20,
                    y - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


    cv2.putText(
        frame,
        f"YOLO FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


    cv2.putText(
        frame,
        f"Objects: {len(detections)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2,
    )


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )


    if not cap.isOpened():

        print(
            "ERROR: Camera gagal dibuka.",
            flush=True,
        )

        return


    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    print(
        f"Camera: "
        f"{actual_width} x "
        f"{actual_height}",
        flush=True,
    )


    print(
        f"MQTT frame: "
        f"{MQTT_TOPIC_FRAME}",
        flush=True,
    )

    print(
        f"MQTT detection: "
        f"{MQTT_TOPIC_DETECTION}",
        flush=True,
    )


    print(
        f"Stream: "
        f"{STREAM_WIDTH} x "
        f"{STREAM_HEIGHT} "
        f"@ {MQTT_FRAME_FPS} FPS",
        flush=True,
    )


    previous_time = (
        time.perf_counter()
    )

    last_frame_publish = 0


    try:

        while True:

            ret, frame = cap.read()


            if not ret:

                print(
                    "Camera read failed",
                    flush=True,
                )

                continue


            original_height, original_width = (
                frame.shape[:2]
            )


            # =================================================
            # YOLO PREPROCESS
            # =================================================

            (
                input_tensor,
                scale,
                pad_x,
                pad_y,
            ) = preprocess(frame)


            # =================================================
            # YOLO INFERENCE
            #
            # TIDAK DIUBAH
            # =================================================

            output = session.run(
                None,
                {
                    input_name:
                        input_tensor
                },
            )


            # =================================================
            # POSTPROCESS
            # =================================================

            detections = postprocess(
                output[0],
                original_width,
                original_height,
                scale,
                pad_x,
                pad_y,
            )


            # =================================================
            # FPS
            # =================================================

            current_time = (
                time.perf_counter()
            )


            fps = 1.0 / max(
                current_time
                - previous_time,
                1e-9,
            )


            previous_time = (
                current_time
            )


            # =================================================
            # DRAW
            # =================================================

            draw_detections(
                frame,
                detections,
                fps,
            )


            # =================================================
            # MQTT DETECTION JSON
            # =================================================

            detection_data = {

                "device": "yolo-stb",

                "timestamp":
                    time.time(),

                "fps":
                    round(
                        fps,
                        2,
                    ),

                "width":
                    original_width,

                "height":
                    original_height,

                "count":
                    len(detections),

                "detections":
                    detections,
            }


            detection_payload = (
                json.dumps(
                    detection_data,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )


            mqtt_client.publish(
                MQTT_TOPIC_DETECTION,
                detection_payload,
                qos=0,
                retain=False,
            )


            # =================================================
            # MQTT JPEG BINARY
            # =================================================

            now = (
                time.perf_counter()
            )


            if (
                now
                - last_frame_publish
                >=
                1.0 / MQTT_FRAME_FPS
            ):

                # ---------------------------------------------
                # Resize HANYA untuk streaming
                #
                # YOLO tetap menggunakan frame asli
                # ---------------------------------------------

                stream_frame = cv2.resize(
                    frame,
                    (
                        STREAM_WIDTH,
                        STREAM_HEIGHT,
                    ),
                    interpolation=cv2.INTER_AREA,
                )


                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ]


                success, jpeg = cv2.imencode(
                    ".jpg",
                    stream_frame,
                    encode_params,
                )


                if success:

                    jpeg_bytes = (
                        jpeg.tobytes()
                    )


                    mqtt_client.publish(
                        MQTT_TOPIC_FRAME,
                        jpeg_bytes,
                        qos=0,
                        retain=False,
                    )


                    last_frame_publish = now


                    frame_size = (
                        len(jpeg_bytes)
                    )


                else:

                    frame_size = 0


            else:

                frame_size = 0


            # =================================================
            # LOG
            # =================================================

            object_summary = ", ".join(
                f'{d["name"]}:1'
                for d in detections
            )


            message = (
                f"FPS: {fps:5.1f} | "
                f"Objects ({len(detections)}): "
                f"{object_summary or 'none'}"
            )


            if frame_size:

                message += (
                    f" | JPEG: "
                    f"{frame_size / 1024:.1f} KB"
                )


            print(
                message,
                flush=True,
            )


    except KeyboardInterrupt:

        print(
            "\nStopping...",
            flush=True,
        )


    finally:

        cap.release()

        mqtt_client.loop_stop()

        mqtt_client.disconnect()

        print(
            "Camera released.",
            flush=True,
        )

        print(
            "MQTT disconnected.",
            flush=True,
        )


if __name__ == "__main__":

    main()