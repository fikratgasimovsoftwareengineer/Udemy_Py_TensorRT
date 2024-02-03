'''
 Pre processing of libraries Stage
    import tensorrt
    import pycuda
    import cv2
    import numpy as np and etc
   
'''
import tensorrt as trt
import pycuda.autoinit

import pycuda.driver as cuda
import cv2

import numpy as np

import os
from PIL import Image

import matplotlib.pyplot as plt


'''
    class name :TRTInference
    INIT: self.logger
    params [1]: engine_path, context
    params [2]: input shape and output shape, class_labels
    
'''
class TRTInference:

    def __init__(self, engine_file_path, input_shape, output_shape, class_labels_file):

        self.logger = trt.Logger(trt.Logger.WARNING)

        self.engine_file_path = engine_file_path

        # load engine 

        self.engine = self.load_engine(self.engine_file_path)

        # init context 
        self.context = self.engine.create_execution_context()


        self.input_shape = input_shape

        self.output_shape = output_shape

        self.class_labels_file = class_labels_file

        # open class file

        with open(class_labels_file, 'r') as class_read:
            self.class_labels = [line.strip() for line in class_read.readlines()]
        
    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine_deserialized = runtime.deserialize_cuda_engine(f.read())
        return engine_deserialized
    

    '''
    param [1] : image path
    results(return) : img_list , img_path
    img resolition : 1, 3 224, 224,
    '''
    def preprocess_img(self, image_path):
        img_list = []

        img_path = []

        for img_original in os.listdir(image_path):
            if img_original.endswith('.jpg') or img_original.endswith('.png') or img_original.endswith('.jpeg'):
                img_full_path = os.path.join(image_path, img_original)

                # open image
                image = Image.open(img_full_path)
                

                # resize img
               # [224, 224]
                img = image.resize((self.input_shape[2], self.input_shape[3]), Image.NEAREST)

                img_np = np.array(img).astype(np.float32) / 255.0

                img_np = img_np.transpose((2,0,1))

                img_np = np.expand_dims(img_np, axis=0)

                img_list.append(img_np)
                

                img_path.append(img_full_path)

        return img_list, img_path
    
    # processing labels with corresponding images
    def postprocess_img(self, outputs):

        # classes 
        classes_indices = []

        for output in outputs:
            class_idx = output.argmax()
            print("Class Detected :", self.class_labels[class_idx])

            classes_indices.append(self.class_labels[class_idx])
        return classes_indices
    
    # inference detection 
    ''' param [0] = self
        param [1] = image_path
        target: Inference Detection on GPU Local
    ''' 
    def inference_detection(self, image_path):
        # list 

        input_list, full_img_paths = self.preprocess_img(image_path)

        results = []

        for inputs, full_img_path in zip(input_list, full_img_paths):
            
            inputs = np.ascontiguousarray(inputs)

            outputs = np.empty(self.output_shape, dtype=np.float32)

            d_inputs = cuda.mem_alloc(1 * inputs.nbytes)

            d_outputs = cuda.mem_alloc(1 * outputs.nbytes)

            bindings = [d_inputs, d_outputs]

            # transfer input to gpu
            cuda.memcpy_htod(d_inputs, inputs)

            # syhrnonize
            self.context.execute_v2(bindings)

            # copy output back to host (cpu)

            cuda.memcpy_dtoh(outputs, d_outputs)

            
            result = self.postprocess_img(outputs)

            d_inputs.free()
            d_outputs.free()

            # results
            results.append(result)

            # display results

            self.display_recognized_images(full_img_path, result)

        return results
    
    '''
        param[0] : image_path
        param[1] : class_label
        target: Displaying and Saving detected images
    '''
    def display_recognized_images(self, image_path, class_label):
        
        image = Image.open(image_path) 

        for class_name in class_label:
            
            # create one directory for detected images
            path_to_detected_imgs = "/tensorfl_vision/Resnet18/images_detected"

            # check path existence

            if not os.path.exists(path_to_detected_imgs):
                os.makedirs(path_to_detected_imgs)

            plt.imshow(image)

            plt.title(f'Recognized Image : {class_name}')

            plt.axis('off')

            save_img = os.path.join(path_to_detected_imgs, f'{class_name}.jpg')

            plt.savefig(save_img)

            plt.close()

            return image
        
engine_file_path = '/tensorfl_vision/Onnx-Inference-Yolov5/resnet.engine'

input_shape = (1,3, 640, 640)

output_shape = (1, 1000)

class_labels = '/tensorfl_vision/Resnet18/imagenet-classes.txt'

path_to_org_images = '/tensorfl_vision/Resnet18/Images'



inference = TRTInference(engine_file_path, input_shape, output_shape, class_labels)

inference.inference_detection(path_to_org_images)