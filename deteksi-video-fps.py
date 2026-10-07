import time

import cv2
from ultralytics import YOLO

# Load the smallest and lightest pretrained YOLO11 model.
model = YOLO("yolo11n.pt")

# Run real-time inference from webcam index 4 and display the current FPS.
previous_time = time.perf_counter()
fps = 0.0

for result in model(
    source=4,
    stream=True,
    imgsz=192,
    conf=0.5,
    max_det=10,
    stream_buffer=False,
    verbose=False,
):
    current_time = time.perf_counter()
    frame_time = current_time - previous_time
    previous_time = current_time

    if frame_time > 0:
        current_fps = 1.0 / frame_time
        fps = current_fps if fps == 0 else (fps * 0.9) + (current_fps * 0.1)

    frame = result.plot(line_width=2)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("YOLO Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()