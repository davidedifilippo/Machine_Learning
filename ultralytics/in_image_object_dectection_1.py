# "YOLO: You Only Look Once: Unified, Real-Time Object Detection".
# "we will use a pre-trained model to detect common object classes like cats and dogs."
from ultralytics import YOLO
import cv2

"The ultralytics package has the YOLO class, used to create neural network models"

model = YOLO("yolov8m.pt")    # "middle-sized model for object detection"

results = model.predict("img.png")
"Per una sola immagine un solo risultato:"
image = cv2.imread("./img.png")


for result in results:
    for box in result.boxes:
        cls = result.names[box.cls[0].item()]
        print("Object type:", cls)
        coords = [round(x) for x in box.xyxy[0].tolist()]
        print("Coordinates:", coords)
        print("Probability:", round(box.conf[0].item(), 2))
        cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0,0,255), 2)
        cv2.putText(image, str(cls), (coords[0], coords[1]-5), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2 )

cv2.imshow("cat_dog", image)
cv2.waitKey(0)
