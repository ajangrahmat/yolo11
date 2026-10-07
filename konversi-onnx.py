from pathlib import Path

import onnx
from ultralytics import YOLO

SOURCE_MODEL = Path("yolo11n.pt")
OUTPUT_MODEL = Path("yolo11n_192.onnx")
IMAGE_SIZE = 192

if not SOURCE_MODEL.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {SOURCE_MODEL}")

# Export YOLO11 nano to ONNX with a 256x256 input.
exported_model = Path(
    YOLO(str(SOURCE_MODEL)).export(
        format="onnx",
        imgsz=IMAGE_SIZE,
        simplify=True,
        opset=12,
    )
)

# Keep the output name explicit and compatible with the installed ONNX Runtime.
onnx_model = onnx.load(str(exported_model))
onnx_model.ir_version = 11
onnx.save(onnx_model, str(OUTPUT_MODEL))

if exported_model != OUTPUT_MODEL and exported_model.exists():
    exported_model.unlink()

print(f"Selesai: {OUTPUT_MODEL}")
print(f"Input model: {IMAGE_SIZE}x{IMAGE_SIZE}")
print("ONNX IR version: 11")
