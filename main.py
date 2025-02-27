import cv2
from ultralytics import YOLO
import math
import numpy as np
from paddleocr import PaddleOCR
from sort import Sort
import os

classnames = ['bien_xe']

model = YOLO("license_plate_detector.pt", task="detect")

prev_frame_time = 0
new_frame_time = 0

last_positions = {}
time_final = 1
tracker = Sort(max_age=30, min_hits=5)

cap = cv2.VideoCapture("data\clip.mp4")

ocr = PaddleOCR(use_angle_cls=True, lang='en')  

if not os.path.exists('save'):
    os.makedirs('save')

# Tạo một tập hợp để lưu ID đã đọc
read_ids = set()
frame_count = 0
wait_frames = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = np.empty((0, 5))

    new_frame_time = cv2.getTickCount()
    frame = cv2.resize(frame, (1080, 720))

    results = model.predict(source=frame, imgsz=640, conf=0.4, verbose=False, vid_stride=5)

    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            objectdetect = classnames[classindex]

            if conf > 40:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                new_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_detections))

    track_result = tracker.update(detections)
    for results in track_result:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Kiểm tra xem ID đã được đọc chưa
        if id not in read_ids:
            cropped_img = frame[y1:y2, x1:x2]
            cropped_img = cv2.resize(cropped_img, (480, 320))
            result = ocr.ocr(cropped_img, cls=True)

            if result is not None and len(result) > 0:
                for line in result:
                    if line:
                        for word_info in line:
                            text = word_info[1][0]
                            if frame_count >= wait_frames:
                                cv2.imwrite(f'save/{text}.png', cropped_img)
                                read_ids.add(id)  
                                frame_count = 0  
                            else:
                                frame_count += 1  
                            cv2.putText(frame, text, (x1 + 2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                print("Không phát hiện văn bản trong ảnh.")

    fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f'FPS: {int(fps)}'
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()