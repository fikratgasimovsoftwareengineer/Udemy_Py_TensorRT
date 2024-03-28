# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import time

base_dir = Path('/tensorfl_vision/YoloV8_OBB')
sys.path.append(str(base_dir))


class YoloV8_OBB_DNN:
    
    
    def __init__(self, conf, nms, score,input_image_path, onnx_model):
        
        # Load class names from data.yaml
        self.CLASSES = ["saldatura"]
        self.colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        
        self.conf = conf
        self.nms = nms
        self.score = score
        self.full_path = base_dir / 'data/batch_images'
        self.input_image_path = input_image_path
        self.onnx_model = onnx_model
        
    def draw_bounding_box(self, img, confidence, left, top, width, height,FPS):
        """
        Draws bounding boxes on the input image based on the provided arguments.
        """
       
        label = "FPS : %.2f , ID: %s, Confidence: %.2f," % (FPS, self.CLASSES[0], confidence)
        print(f"Label is {label}")
       
        cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 5)

                    # task for learnign put text 
        cv2.putText(img, label, (left, top + 20), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.namedWindow('detection.jpg', cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow('detection.jpg', 640, 640)
        
        cv2.imshow('detection.jpg', img)

        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    
    def __call__(self):    
        for img in os.listdir(self.input_image_path):
            if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):
                img_full_path = os.path.join(self.full_path, img)
                original_image = cv2.imread(img_full_path)
                self.infernence_obb(self.onnx_model, original_image)
                

    def infernence_obb(self, onnx_model, original_image):
        """
        Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.
        """
       

        # Load the ONNX model
        model = cv2.dnn.readNetFromONNX(onnx_model)
    
        # config gpu    
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # gpu 
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

       

        # Read the input image
    
        start_time = time.time()
      
        
        blob = cv2.dnn.blobFromImage(original_image, scalefactor=1/255.0,  size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)

            # perform inference
        outputs = model.forward()
        
        end_time = time.time()
        
        # Frame per second

        FPS = 1.0 * (end_time-start_time)
            
            
        height, width = original_image.shape[:2]

            # x_scale

        x_scale = width / 640
        y_scale = height / 640
      

        bboxes = []
    
        confidences = []
        class_ids = []
            

        # n_detections = outputs[0].shape[1]
        #[1,6,8400]
        # [x,y,width, height, conficence, class_id]
        # Process outputs
        for i in range(outputs.shape[2]): # Iterate through each detection
            detection = outputs[0, :, i]
            # extract cx, cy, width and heigth from model
            cx, cy, w, h, conf, class_id = detection[:6]
                #x, y, w, h, conf, class_id = detection
            if conf > 0.2:
                confidences.append( conf)
                
                
                left = int((cx - w / 2) *x_scale)

                top = int((cy- h / 2 ) *y_scale )

                width = int(w * x_scale)

                height = int(h * y_scale)
                box = np.array([left, top,width, height])
                    
                bboxes.append(box)
                    
    
    
    
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.conf, self.nms)

        for i in indices:
            box = bboxes[i]

            left = box[0]

            top = box[1]

            width = box[2]

            height = box[3]

                    
            self.draw_bounding_box(original_image, conf, left, top, width, height,FPS)
                    #x, y, x_plus_w, y_plus_h = (detection[3:7] * np.array([width, height, width, height])).astype(int)
                    #draw_bounding_box(original_image, class_id, confidence, x, y, x_plus_w, y_plus_h)
        

def main():
    input_image_path = "/tensorfl_vision/YoloV8_OBB/data/batch_images"  # Update this path
    onnx_model_path = "/tensorfl_vision/YoloV8_OBB/data/best.onnx"  # Update this path
    model_obb = YoloV8_OBB_DNN(0.15, 0.25, 0.4,input_image_path,onnx_model_path)
    model_obb()
    
if __name__ == "__main__":
    main()