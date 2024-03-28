'''
    Date: 23/07/2023
    Project : yolov5 Inference on video frames
    developed by : Fikrat Gasimov

'''



# import preprocessing libraries

import tensorrt as  trt
import pycuda.autoinit
import cv2
import pycuda.driver as cuda

import numpy as np

import os
from PIL import Image
import matplotlib.pyplot as plt

import yaml
import time


'''
    Class Name: yolov5TensorRT
    target: INIT Class params
    param[1]: engine_file_path
    param[2]: input_shape
    param[3]: output_shape
    param[4]: classes_label_file
    param[5]: conf_threshold
    param[6] : score_threshold
    param[7] : nms_threshold

'''

class yolov5TensorRT:
   
    def __init__ (self, engine_file_path, input_shape, output_shape, img_path, conf_threshold, score_threshold, nms_threshold):
        
        # Warning while engien loading
        self.logger = trt.Logger(trt.Logger.WARNING) # ?
        
        # INIt Params
        self.engine_file_path = engine_file_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.img_path = img_path
       
        self.conf_threshold = conf_threshold
        self.score_threhold = score_threshold
        self.nms_threshold = nms_threshold
        
        self.CLASSES = ["saldatura"]
        
        # load engine
        self.engine = self.load_engine(self.engine_file_path)
        self.context = self.engine.create_execution_context()
        
        self.full_path = '/tensorfl_vision/YoloV8_OBB/data/batch_images'
        
         
    def draw_bounding_box(self, img, confidence, left, top, width, height):
        """
        Draws bounding boxes on the input image based on the provided arguments.
        """
       
        label = "ID: %s, Confidence: %.2f," % ( self.CLASSES[0], confidence)
        print(f"Label is {label}")
       
        cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 5)

                    # task for learnign put text 
        cv2.putText(img, label, (left, top + 20), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.namedWindow('detection.jpg', cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow('detection.jpg', 800, 1000)
        
        cv2.imshow('detection.jpg', img)

        cv2.waitKey(4000)
        cv2.destroyAllWindows()
        
    def __call__(self):    
        for img in os.listdir(self.img_path ):
            if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):
                
                img_full_path = os.path.join(self.full_path, img)
                
                image_name = os.path.splitext(os.path.basename(img_full_path))[0]
                print(image_name)
                
                original_image = cv2.imread(img_full_path)
                
                self.org_frame_h, self.org_frame_w = original_image.shape[:2]
                
                img_resized = cv2.resize(original_image, (self.input_shape[2], self.input_shape[3]), interpolation=cv2.INTER_AREA)
        
                self.resized_frame_h, self.resized_frame_w = img_resized.shape[:2]
        
                # convert to n umpy and divide it by float32 bit and 255.0 
               
                img_np = np.array(img_resized).astype(np.float32) / 255.0
                # convert channel [3, 640, 640]
                img_np = np.transpose(img_np, (2, 0, 1))
                
                
                self.inference_detection(original_image, img_np)
    ''' 
        loading engine file and deserialize for an inference
        param[1]: engine_file_path
        param[out]: deserialized_engine
    '''
    def load_engine(self,engine_file_path):
        with open(engine_file_path, 'rb') as f:
            
            runtime = trt.Runtime(self.logger)
            engine_deserialized = runtime.deserialize_cuda_engine(f.read())
            
        return engine_deserialized

        
    '''
        param[1]: video path
        param[out1]:frame
        param[out2]:outputs 
    '''
    def inference_detection(self, image, img_np):
        
        self.total_time = 0
        
        self.num_frames = 0
    
        
        self.num_frames += 1
        
        self.start = time.time()
        
        # pass array
        inputs = np.ascontiguousarray(img_np)
        
        # outputs shape = [1, 6, 8400]
        outputs = np.empty(self.output_shape, dtype=np.float32)
        
        
        d_inputs = cuda.mem_alloc(1 * inputs.nbytes)
        
        d_outputs = cuda.mem_alloc(1 * outputs.nbytes)

        bindings = [d_inputs, d_outputs]
        
        cuda.memcpy_htod(d_inputs, inputs)
        
        self.context.execute_v2(bindings)
        
        cuda.memcpy_dtoh(outputs, d_outputs)
        
        d_inputs.free()
        d_outputs.free()
        
        # end time 
        
        self.end = time.time()
        
        self.total_time += (self.end - self.start)
        
        self.FPS = self.num_frames / self.total_time
        
        # post processing gpu results
        
        self.postprocessing_recognized_frames(image, outputs)


        
        return outputs
        
        
    '''
    target: postprocessing
    param[1]: frame
    param[2]: yolov5 output gpu results
    '''    
    def postprocessing_recognized_frames(self, frame, yolov8_output_obb):
        
        
        detections = yolov8_output_obb.shape[2]
        
        width, height = frame.shape[:2]
        
        x_scale = self.org_frame_w / self.resized_frame_w
        y_scale = self.org_frame_h / self.resized_frame_h
        
        conf_threshold = self.conf_threshold
        
    
        
        nms_threshold= self.nms_threshold
        
        class_ids = []
        
        confidences = []
        
        bboxes = []
        
        
        for i in range(detections):
            
            detection = yolov8_output_obb[0, :, i]
            
            cx, cy, w, h, conf, class_id = detection[:6]
            
            if conf > 0.2:
                confidences.append( conf)
                
                
                left = int((cx - w / 2) *x_scale)

                top = int((cy- h / 2 ) *y_scale )

                width = int(w * x_scale)

                height = int(h * y_scale)
                box = np.array([left, top,width, height])
                    
                bboxes.append(box)
                
                    
        
        indices_nonmax = cv2.dnn.NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold)
        
        for i in indices_nonmax:
            box = bboxes[i]
            left = box[0]
            top = box[1]
            width = box[2]  
            height = box[3]
            
            self.draw_bounding_box(frame, conf, left, top, width, height)
          
            
            
def main():
    engine_file_path =  '/tensorfl_vision/YoloV8_OBB/data/best.engine'
    input_shape = (1, 3, 640, 640)
    output_shape = (1, 6, 8400)
    img_path = '/tensorfl_vision/YoloV8_OBB/data/batch_images'

    
    inference = yolov5TensorRT(engine_file_path, input_shape, output_shape, img_path, 0.18, 0.45, 0.35)
    
    inference()
    
    
if __name__=="__main__":
    main()
    
    