import math
import time
import cv2
import numpy as np

import onnxruntime
import sys
from pathlib import Path

base_dir = Path("/tensorfl_vision/YoloV8ONNXInference")
sys.path.append(str(base_dir))

from transform.utils import xywh2xyxy, nms, draw_detections, sigmoid

'''

CLASS INIT


'''

class YoloSeg:
    
    def __init__(self, path, conf_thres = 0.7, ios_thres = 0.5, num_mask=32):
        
        self.path= path
        self.conf_thres = conf_thres
        self.ios_thres = ios_thres
        
        self.num_mask = num_mask
        
        
        self.initialize_model(path) # added 
        
    def __call__(self, image):
        return self.segment_objects(image)
    
    def initialize_model(self, path):
        
        # . In the example below if there is a kernel in the CUDA execution provider ONNX Runtime 
        # executes that on GPU. If not the kernel is executed on CPU.
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        
        # self get_input detais
        self.get_input_details()
        
        # self output details
        self.get_output_details()
        
    def segment_objects(self, image):
        
        input_tensor = self.prepare_input(image)
        
        
        # inference
        
        outputs = self.inference(input_tensor) 
        
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        
        # create map
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])
        
        return self.boxes, self.scores, self.class_ids, self.mask_maps
    
    
    def inference(self, input_tensor):
        start = time.time()
        # ONNX SESSION RUN 
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        
        end = time.time()
    
        print(f"Inference time elaps : {start-end} in seconds")
        
        return outputs
    def get_input_details(self):
        
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        
        self.input_shape=model_inputs[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]
        
    def get_output_details(self):
        
        model_outputs = self.session.get_outputs()
        
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        
        
    def prepare_input(self, image):
        
        self.img_height, self.img_width = image.shape[:2]
        
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # resize input
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        
        # scale it to pixel [0,1]
        
        input_img = input_img / 255.0
        
        input_img = input_img.transpose(2, 0, 1) # transpose
        
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        
        return input_tensor
    

    def process_box_output(self, box_output):
        
        # take output 0 , and preprocess on 1, 116, 8400 
        # remove 1 and transpose to 8400, 116
        
        predictions = np.squeeze(box_output).T
        
        # num of mask 32 for yolov8
        
        num_classes = box_output.shape[1] - self.num_mask - 4
        
        # filtering out object scores and confidense,threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        
        # scores from 8400
        predictions = predictions[scores > self.conf_thres, :]
        
        scores = scores[scores > self.conf_thres]
        
        if len(scores) == 0:
            
            return [],[],[],np.array([])
        
        box_predictions = predictions[...,:num_classes+4]
        mask_predictions = predictions[...,num_classes+4:]
        
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        
        # get bounding for each object
        boxes = self.extract_boxes(box_predictions) 
        
        # apply non max supression
        indices = nms(boxes, scores, self.ios_thres)
        
        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]
        
        
    def process_mask_output(self, mask_predictions, mask_output):
        
        if mask_predictions.shape[0] == 0:
            return []
        
        mask_output = np.squeeze(mask_output)
        
        # calculate mask maps for each box
        
        num_mask, mask_height, mask_width = mask_output.shape # CHW
        
        
        masks = sigmoid(mask_predictions @ mask_output.reshape(num_mask, -1))
        
        masks = masks.reshape((-1, mask_height, mask_width))
        
        # downscale boxes to match mask size
        
        scale_boxes = self.rescale_boxes(self.boxes,
                                         (self.img_height, self.img_width),
                                         (mask_height, mask_width))
        
        # get mask map for every box / mask pair
        
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        
        for i in range(len(scale_boxes)):
            
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))
            
            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))
            
            
            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            
            crop_mask = cv2.resize(scale_crop_mask,
                                   (x2-x1, y2-y1),
                                   interpolation=cv2.INTER_CUBIC)
            
            crop_mask = cv2.blur(crop_mask, blur_size)
            
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask
            
        return mask_maps    
   
    def extract_boxes(self, box_predictions):
        
        boxes = box_predictions[:, :4]
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))
        
        # convert boxes to xyxyx format
        boxes = xywh2xyxy(boxes)
        
        
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)
        
        return boxes
        
    
    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # scale factor
        
        scale_x = image_shape[1] / input_shape[1]
        scale_y = image_shape[0] / input_shape[0]
        
        # rescale boxes
        rescaled_boxes = boxes.copy()
        
        #rescale xmin,ymin, xmax, ymax
        
        rescaled_boxes[:, 0] = boxes[:,0] * scale_x
        rescaled_boxes[:, 1] = boxes[:,1] * scale_y
        rescaled_boxes[:, 2] = boxes[:,2] * scale_x
        rescaled_boxes[:, 3] = boxes[:,3] * scale_y
        
        return rescaled_boxes
        
        
        
    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores, 
                               self.class_ids, mask_alpha)
        
    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores, 
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)
        
        
if __name__ == '__main__':
    
    from imread_from_url import imread_from_url
    
    model_path = base_dir / 'data/input/yolov8m-seg.onnx'
    
    yoloseg = YoloSeg(model_path, conf_thres=0.3, ios_thres=0.5)
    
    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    
    img = imread_from_url(img_url)
    
    yoloseg(img)
    
    
    combined_img = yoloseg.draw_masks(img)
    
    cv2.namedWindow("Detect&Segment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detect&Segment", 1000,720)
    cv2.imshow("Detect&Segment", combined_img)
    
    
    while True:
        
        k = cv2.waitKey(30) & 0xFF
        if k==27:
            cv2.destroyAllWindows()