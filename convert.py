from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r"license_plate_detector.pt")

# model.export(format="engine",half = True, simplify=True, imgsz=640)

# # # # # Run inference on 'bus.jpg' with arguments
model.predict(source="data\car1.jpg", imgsz=640,conf = 0.65, show = True, save = True)

#xem cách lớp được nhận dạng
# print(model.names)