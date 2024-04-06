from ultralytics import YOLO
import cvzone
import cv2
import math
import time

model = YOLO("./models/best2.pt")

classnames = ["Fire", "Smoke"]

# cap = cv2.VideoCapture(0) # webcam
cap = cv2.VideoCapture("./test_content/vids/fire2.mp4")

prev_frame_time = 0
new_frame_time = 0

while (cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame,(1280,720))
    result = model(frame,stream=True)
    font = cv2.FONT_HERSHEY_DUPLEX

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            classname = int(box.cls[0])
            if confidence > 10:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 0, 255), 3)
                cvzone.putTextRect(frame, f"{classnames[classname].upper()} {confidence}%",
                                   [x1 + 8, y1 + 100],
                                   scale=2,thickness=2)

    new_frame_time = time.time()

    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps = f"fps {str(int(fps))}"

    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("frame", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break

cap.release()
cv2.destroyAllWindows()
