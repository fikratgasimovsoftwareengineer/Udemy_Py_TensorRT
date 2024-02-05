from ultralytics import YOLO


# catch and try statement for better error handling

try:
    model = YOLO("yolov8m-seg.pt")
    model.export(format="onnx")
    
except Exception as e:
    print(f"Model is not converted or exported as ONNX , due to error {e}")