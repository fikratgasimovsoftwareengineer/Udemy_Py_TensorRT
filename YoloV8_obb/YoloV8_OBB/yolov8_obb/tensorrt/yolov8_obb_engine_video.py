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
   
    def __init__ (self, engine_file_path, input_shape, output_shape, conf_threshold, score_threshold, nms_threshold):
        
        # Warning while engien loading
        self.logger = trt.Logger(trt.Logger.WARNING) # ?
        
        # INIt Params
        self.engine_file_path = engine_file_path
        self.input_shape = input_shape
        self.output_shape = output_shape
       
        self.conf_threshold = conf_threshold
        self.score_threhold = score_threshold
        self.nms_threshold = nms_threshold
        
        self.CLASSES = ["saldatura"]
        
        # load engine
        self.engine = self.load_engine(self.engine_file_path)
        self.context = self.engine.create_execution_context()
        
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
        target:preprocessing video frames
        param[1]: video_path
        param[out]: video frames
    '''
    
    def preprocess_video(self,video_path):
        
        video = cv2.VideoCapture(video_path)
       
     

        # while opens, do preprocessing
        while video.isOpened():
           
            
            ret, frame = video.read()
          
            if not ret:
                
                break
            
           
            frame = cv2.resize(frame, (800, 500))
                
            self.org_frame_h, self.org_frame_w = frame.shape[:2]
            
            
            '''
                NORMALIZATION FOR INFERENCE
            '''
            img_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[3]), interpolation=cv2.INTER_AREA)
            
            self.resized_frame_h, self.resized_frame_w = img_resized.shape[:2]
            
            # convert to n umpy and divide it by float32 bit and 255.0
            
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            
            img_np = np.transpose(img_np, (2, 0, 1))
            
            yield img_np, frame
       
        
        video.release()
        
    '''
        param[1]: video path
        param[out1]:frame
        param[out2]:outputs 
    '''
    def inference_detection(self,video_path):
        
        self.total_time = 0
        
        self.num_frames = 0
        
        
        for inputs, frame in self.preprocess_video(video_path):
            
            self.num_frames += 1
            
            self.start = time.time()
            
            inputs = np.ascontiguousarray(inputs)
            
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
            
            self.postprocessing_recognized_frames(frame, outputs)


        
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
        
        score_threshold = self.score_threhold
        
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
            
            
            label = "{:.2f}, FPS: {:.2f}".format(confidences[i], self.FPS)
            
            cv2.rectangle(frame, (left, top), (left + width, top+height), (0, 255, 0), 3)
            cv2.putText(frame, label, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            
            cv2.imshow('Detection.jpg', frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows() 
            
            
def main():
    engine_file_path =  '/tensorfl_vision/YoloV8_OBB/data/best.engine'
    input_shape = (1, 3, 640, 640)
    output_shape = (1, 6, 8400)
    video_path = '/tensorfl_vision/YoloV8_OBB/data/video2_cut.mp4'

    
    inference = yolov5TensorRT(engine_file_path, input_shape, output_shape, 0.4, 0.45, 0.35)
    
    inference.inference_detection(video_path)
    
    
if __name__=="__main__":
    main()
    
    