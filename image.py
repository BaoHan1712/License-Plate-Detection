import cv2
from ultralytics import YOLO
import math
import numpy as np
from paddleocr import PaddleOCR

classnames = ['bien_xe']

model = YOLO("license_plate_detector.pt", task="detect")

prev_frame_time = 0
new_frame_time = 0

last_positions = {}
time_final = 1


image_path = 'data\car_blur.png' 
frame = cv2.imread(image_path)


ocr = PaddleOCR(use_angle_cls=True, lang='en')


if frame is not None:
    detections = np.empty((0, 5))

    new_frame_time = cv2.getTickCount()
    frame = cv2.resize(frame, (1080, 720))

    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False, max_det=6)

    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            objectdetect = classnames[classindex]
            
            if conf > 50:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                new_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_detections))

                cropped_img = frame[y1:y2, x1:x2]

                cropped_img = cv2.resize(cropped_img, (480, 320))  
                result = ocr.ocr(cropped_img, cls=True)

                if result is not None:
                    for line in result:
                        for word_info in line:
                            text = word_info[1][0]  
                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imshow('Object Detection', frame)
    cv2.imshow('Cropped Image',cropped_img)
    cv2.waitKey(0) 
else:
    print("Không thể đọc ảnh từ đường dẫn đã chỉ định.")