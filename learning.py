from ultralytics import YOLO

model = YOLO('yolov8n.yaml') 
data_path = dataset.location + '/data.yaml' 

model.train(data=data_path, epochs=100, imgsz=640)
