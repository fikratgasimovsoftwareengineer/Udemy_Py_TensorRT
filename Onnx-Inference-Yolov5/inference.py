import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import time
class TRTInference:
        
    # specify engine file path and input and output shape
    def __init__(self, engine_file_path, input_shape, output_shape, class_labels_file,conf_threshold, score_threshold, nms_threshold):
        self.logger = trt.Logger(trt.Logger.WARNING)

        self.engine_file_path = engine_file_path
            
        ## load engine here
        self.engine = self.load_engine(self.engine_file_path)

        # craete context
        self.context = self.engine.create_execution_context()

        self.conf_threshold = conf_threshold

        self.score_threshold = score_threshold

        self.nms_threshold = nms_threshold

        # input shape
        self.input_shape = input_shape
            
        self.class_labels_file =  class_labels_file
        self.count = 0
            
        # output shape
        self.output_shape = output_shape
          
       # with open(class_labels_file, 'r') as class_read:
          #  self.class_labels = [line.strip() for line in class_read.readlines()]
        with open(class_labels_file, 'r') as class_read:
            
            data = yaml.safe_load(class_read)
            self.class_labels = [name for name in data['names'].values()]

        

    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine_desentriliazed = runtime.deserialize_cuda_engine(f.read())

            return engine_desentriliazed
    
        
    def preprocess_image(self, image_path):
        
         #img list
        img_list = []
        # img path
        img_path = []
        count = 0
       
        for img_original in os.listdir(image_path):
          
            if img_original.endswith('.jpg') or img_original.endswith('.png') or img_original.endswith('jpeg'):
                img_full_path = os.path.join(image_path, img_original)
                
                self.img = cv2.imread(img_full_path)

                self.org_h, self.org_w = self.img.shape[:2]
           
                #img size = [640, 640]
                
                self.img_resized = cv2.resize(self.img, (self.input_shape[2], self.input_shape[3]), interpolation=cv2.INTER_AREA)
                #self.img_resized = self.resize_with_aspect_ratio(self.img, target_size=(self.input_shape[2], self.input_shape[3]))
                img_np = np.array(self.img_resized).astype(np.float32) / 255.0

                img_np = img_np.transpose((2,0,1))

                img_np = np.expand_dims(img_np, axis=0)

                self.resized_imgH, self.resized_imgW =self.img_resized.shape[:2]               

                count +=1
                
                img_list.append(img_np)
                

                img_path.append(img_full_path)

                if count >= 12:
                    continue

                # call detection 
               
        return  img_list, img_path
        
    
    def inference_detection(self,image_path):
        
        input_list, full_img_paths = self.preprocess_image(image_path)

        results = []
        stream = cuda.Stream()

        self.total_time = 0
        self.num_frames = len(input_list)

        for inputs ,full_img_path in zip(input_list, full_img_paths):

            # start time
            self.start = time.time()

            inputs = np.ascontiguousarray(inputs)

            outputs = np.empty(self.output_shape, dtype=np.float32)
          

            d_inputs = cuda.mem_alloc(1 * inputs.nbytes)

            d_outpus = cuda.mem_alloc(1 * outputs.nbytes)

            bindings = [d_inputs ,d_outpus]
      
            
            '''cuda.memcpy_htod_async(d_inputs, inputs, stream)

            self.context.execute_async(bindings=bindings, stream_handle=stream.handle)

            cuda.memcpy_dtoh_async(outputs, d_outpus, stream)

            stream.synchronize()'''

          
            cuda.memcpy_htod(d_inputs, inputs)
        

            self.context.execute_v2(bindings)

            # copy output back to host
            cuda.memcpy_dtoh(outputs,d_outpus)

             #cuda.memcpy_htod_async(d_outpus, outputs, stream)
           # result = self.postprocess_img(outputs)      
         

            d_inputs.free()

            d_outpus.free()

            # end time
            self.end = time.time()
            self.total_time  += (self.end- self.start)

            self.fps = self.num_frames / self.total_time
            self.postprocess_recognized_image(full_img_path, outputs)    

        return outputs
        
    # save images with detected results
    def postprocess_recognized_image(self, image_path, yolov5_output):
        
        #image = Image.open(image_path)
        image = cv2.imread(image_path)
       # img = image.cop()
       
    
        #for class_name in class_label:
        detections = yolov5_output[0].shape[0]

    #    print(yolov5_output[0][0][0])

        width, height =  image.shape[:2]

        # re-scaling
        x_scale = width / self.resized_imgW
        y_scale = height / self.resized_imgH

        #width, height =  self.img.shape[:2]

      

        conf_threshold = self.conf_threshold
        score_threshold = self.score_threshold
        nms_threshold = self.nms_threshold

        class_ids = []
        confidences = []
        bboxes = []
        #print(yolov5_output)
    

        for i in range(detections):

            detect = yolov5_output[0][i]
            #print(detect)
            
            getConf = detect[4]
          
            if getConf >= conf_threshold:

            
                class_score = detect[5:]  

                class_idx = np.argmax(class_score)

                if (class_score[class_idx] > score_threshold):

                    # confidence
                    confidences.append(getConf)
                    
                    class_ids.append(class_idx)

                    #get center and w,h coordinates
                    cx, cy, w, h = detect[0], detect[1], detect[2], detect[3]
                   # print("Center X",cx, "Center Y ", cy, " Width", w, "Height: ", h)
                   # print('*********************************************************')
                    #print('\n')
                    

                    # left
                    left = int((cx - w/2) * x_scale)

                    # top
                    top = int((cy - h/2) * y_scale)

                    # width
                    width = int(w * x_scale)

                    #height
                    height = int(h * y_scale) 
                    
                    # box
                    box = np.array([left, top, width, height])

                    # bboxes
                    bboxes.append(box)
                    
       # print("output of box")
       # print(bboxes)
        # get max suppresion
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
        
            box = bboxes[i]
            left = box[0] 
            top = box[1] 
            width = box[2] 
            height = box[3]

           
            print("Box Left ",left , "Box Top ", top, "Box Width ", width, "Box height: ", height )
            print('*********************************************************')
            print('\n')
        
            print(self.class_labels[class_ids[i]])
            print()
            label = "{}:{:.2f}".format(self.class_labels[class_ids[i]], confidences[i])

            
            #label2 = "FPS:".format(self.fps)

            cv2.rectangle(image, (left, top),(left + width, top + height), (0,255,0),3)

            cv2.putText(image, label, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
           

        cv2.namedWindow('result.jpg', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result.jpg', 900, 800)
        cv2.imshow('result.jpg', image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
     
        
        #return image, save_img
#engien path 


def main():

    engine_file_path ='/tensorfl_vision/Onnx-Inference-Yolov5/models_engine/yolov5s.engine'

    # Load the TensorRT engine
    input_shape = (1,3, 640, 640)
   # output_shape = (1, 25500, 7)
    output_shape = (1, 25200, 85)

    #image_path = '/deeplearning/resnet/rose.jpeg'



    image_path = '/tensorfl_vision/Onnx-Inference-Yolov5/Images'


    path_to_class = "/tensorfl_vision/Onnx-Inference-Yolov5/coco.yaml"



    inference = TRTInference(engine_file_path, input_shape, output_shape, path_to_class, 0.4, 0.45, 0.35)

    inference.inference_detection(image_path)

if __name__ == "__main__":
    main()