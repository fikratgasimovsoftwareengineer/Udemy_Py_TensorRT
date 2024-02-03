'''
Packages:
    cv2
    numpy
    time
    os

'''

import os
import cv2
import numpy as np
import time
import os


'''
    Target: Making Inference With Yolov5 Onnx model on Set of Images
    Name: DetectV5Onnx
    param[1]: imgs_path
    param[2]: model path
    param[3]: img_height
    param[4]: img_width
    param[5]: confidence threshold
    param[6]: score threshold
    param[7]: non max suppression threshold
    param[8]: classes path 

'''

class  DetectV5Onnx:

    # Init Declaration

    def __init__(self, imgs_path, model_path, imgs_width, imgs_height, conf_threshold, score_threshold, nms_threshold, classes_path):

        # Declare params

        self.imgs_path = imgs_path

        #
        self.model_path = model_path

        #
        self.imgs_width = imgs_width

        #
        self.imgs_height = imgs_height

        #
        self.conf_threshold = conf_threshold

        self.score_threshold = score_threshold

        self.nms_threshold = nms_threshold

        self.classes_path = classes_path

    # python prebuilt call function

    '''
        target: Load images and Onnx Model
        Return Type : OUT[]
    '''
    def __call__(self):

        for img in os.listdir(self.imgs_path):
            if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png'):
                # full absolute path to image
                img_full_path = os.path.join(self.imgs_path, img)

                # read image 

                image = cv2.imread(img_full_path)

                # load onnx model

                network = cv2.dnn.readNetFromONNX(self.model_path)

                # gpu 
                network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

                # config gpu    
                network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                # read classess

                classes = self.class_name()
                
                # detection function is called!
                self.detection(image, network, classes)


    # read labels of coco 
    '''
        target: read class labels
        Out: [classes list]
    '''
    def class_name(self):
        
        # list of classes

        classes = []

        file = open(self.classes_path, 'r')

        while True:
            name = file.readline().strip('\n')
            classes.append(name)

            if not name:
                break
        return classes
    

    '''

    target: To make Inference and SHow Image
    name : detection
    param[1]: image
    param[2]: network
    param[3]: classes

    '''

    def detection(self, img, net, classes):

        # blob to apply to input image
        #Out[] : Returnr 4-dimensional Mat with NCHW dimensions order. 

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, mean=(0,0), crop=False)

        # set to input model
        #Sets the new input value for the network
        net.setInput(blob)

        # start time

        t1 = time.time()

        # Unconnected Layer by Index
        output_layers = net.getUnconnectedOutLayersNames()

        #Out[]:blob for first output of specified layer.
        # detection results
        outputs = net.forward(output_layers)

        t2 = time.time()

        print('OpenCV DNN YOLOV5 Inference time :' , t2- t1)

        # number of detections 
        # 25200
        n_detections = outputs[0].shape[1]

        height, width = img.shape[:2]

        # x_scale

        x_scale = width / self.imgs_width
        y_scale = height / self.imgs_height

        
        # confidence
        confidence_threshold = self.conf_threshold

        # score
        score_threshold = self.score_threshold

        # non max suppresion 
        nms_threshold = self.nms_threshold

        # lsit of class ids

        class_ids = []

        confidences = []

        bboxes = []


        # loop through detections

        for i in range(n_detections):
            #detect 
            detect = outputs[0][0][i]

            confidence = detect[4]

            if confidence >= confidence_threshold:
                class_score = detect[5:]

                class_id = np.argmax(class_score)

                if (class_score[class_id] > score_threshold):
                    
                    confidences.append(confidence)

                    class_ids.append(class_id)

                    cx, cy, w, h = detect[0], detect[1], detect[2], detect[3]

                    # calculate Bounding box coordinates

                    left = int((cx - w / 2) *x_scale)

                    top = int((cy- h / 2 ) *y_scale )

                    width = int(w * x_scale)

                    height = int(h * y_scale)

                    box = np.array([left, top, width, height])

                    bboxes.append(box)

            # non max suppression 

        indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)

        for i in indices:
            box = bboxes[i]

            left = box[0]

            top = box[1]

            width = box[2]

            height = box[3]

            # label

            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            
                # rectangle

            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 3)

                    # task for learnign put text 
            cv2.putText(img, label, (left, top + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.namedWindow('detection.jpg', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('detection.jpg', 900, 800)
        cv2.imshow('detection.jpg', img)

        cv2.waitKey(4000)
        cv2.destroyAllWindows()



def main():
    image_path = '/tensorfl_vision/Onnx_YoloV5/Images'

    onnx_path = '/tensorfl_vision/Onnx_YoloV5/Models/yolov5s.onnx'

    classes_path = '/tensorfl_vision/Onnx_YoloV5/coco-classes.txt'

    
    instance = DetectV5Onnx(image_path, onnx_path, 640, 640, 0.34, 0.38, 0.3, classes_path)

    instance()


if __name__== "__main__":
    main()
















