from ultralytics import YOLO
model = YOLO("yolo11n.pt")

results = model('gambar.jpg')

for result in results:
    result.show()
    