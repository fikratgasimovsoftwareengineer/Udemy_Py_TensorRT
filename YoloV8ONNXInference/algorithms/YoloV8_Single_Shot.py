# preprocessing import file

import cv2
import sys
from pathlib import Path
import argparse
import os
from ultralytics import YOLO


base_dir = Path('/tensorfl_vision/YoloV8ONNXInference/')
sys.path.append(str(base_dir))


# Class Declare

class YoloV8SingleShot:
    
    
    # model path
    def __init__(self, images_path, output_directory):
       
        self.images_path = images_path
        self.output_directory = output_directory
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    # Class Method Init
    def inference(self):
        
        for images in os.listdir(self.images_path):
            img = os.path.join(self.images_path, images)
            img = cv2.imread(img)
            
            self.model = YOLO(base_dir / "data/input/yolov8m-seg.pt")
            
            self.model.predict(source=img, conf=0.25, save=True, name=self.output_directory)
            

    
# Class INIT
def main():
    
    parser = argparse.ArgumentParser(description='Yolov8 Single Shot Inference on CPU')
    
    parser.add_argument('--images_path', type=str, required=True, help='Path to Images directory')
    parser.add_argument('--output_directory', type=str, required=True, help='Output directory parsing')
    
    args=parser.parse_args()
    yolov8 = YoloV8SingleShot(args.images_path, args.output_directory)
    
    yolov8.inference()
    
    
if __name__ == '__main__':
    main()