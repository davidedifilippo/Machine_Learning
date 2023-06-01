from ultralytics import YOLO
import numpy as np
import torch
import cv2

cap = cv2.VideoCapture(1)

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if torch.backends.mps.is_available():
        results = model(frame, device="mps")
    else:
        results = model(frame, device="cpu")

    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    print(bboxes)
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
    cv2.imshow("video", frame)

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()
