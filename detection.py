import cv2
import random
from google.colab.patches import cv2_imshow
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt') 

cap = cv2.VideoCapture('test.mp4') 

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

frame_count = 0
display_interval = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device='cpu') 

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

    if frame_count % display_interval == 0:
        cv2_imshow(frame)

    frame_count += 1

cap.release()
out.release()
