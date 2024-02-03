
import cv2
import numpy as np
import argparse
import time
import os


class detectv5:
    def __init__(self, img_path, model, imgs_w, imgs_h, conf_threshold, score_threshold, nms_threshold, classes_txt):
        self.conf= conf_threshold
        self.score=score_threshold
        self.nms=nms_threshold
        self.img_path= img_path
        self.model= model
        self.img_w= imgs_w
        self.img_h = imgs_h
        self.classes_file= classes_txt

    # instance will be called in a way of function
    def __call__(self):
        
        for img in os.listdir(self.img_path):
            if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png'):
                img_full_path = os.path.join(self.img_path, img) 
                img = cv2.imread(img_full_path)
                net = cv2.dnn.readNetFromONNX(self.model)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                classes= self.class_name() # class name
                self.detection(img, net, classes) # call detection
        

    def class_name(self):
        classes=[]
        file= open(self.classes_file,'r')
        while True:
            name=file.readline().strip('\n')
            classes.append(name)
            if not name:
                break
        return classes

    def detection(self, img, net, classes): 
        
        
        '''
        Mat cv::dnn::blobFromImage 	( InputArray  	image,
        double  	scalefactor = 1.0,
        const Size &  	size = Size(),
        const Scalar &  	mean = Scalar(),
        bool  swapRB = false,
        bool  crop = false,
        int    ddepth = CV_32F 
        ) 		
        '''
        
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640,640), swapRB=True, mean=(0,0,0), crop= False)
        # set Input value for network
        net.setInput(blob)

        # start time
        t1= time.time()

        # forward :: Runs forward pass to compute output of layer with name outputName. 

        # Return blob for first output of specified layer.

        # By default runs forward pass for the whole network.
        #         
        output_layers = net.getUnconnectedOutLayersNames()
        outputs= net.forward(output_layers) #getUnconnectedOutLayersNames() : Returns names of layers with unconnected outputs. 
       # outputs = net.forward()
        #end time
        t2 = time.time()

        
        #(out.shape)
        print('Opencv dnn yolov5 inference time: ', t2- t1)

      
       

        n_detections= outputs[0].shape[1]  
        
        height, width= img.shape[:2]

        # scale x and y
        x_scale= width/self.img_w
        y_scale= height/self.img_h
        

        conf_threshold= self.conf

        score_threshold= self.score

        nms_threshold=self.nms

        class_ids=[]
        confidences=[]
        bboxes=[]

        
      
        for i in range(n_detections):

            detect=outputs[0][0][i] 

            confidence= detect[4]
            #print(f"Detection {i}: confidence {confidence}")

            if confidence >= conf_threshold:
                class_score= detect[5:]
              

                class_id= np.argmax(class_score)
               
                
                if (class_score[class_id]> score_threshold):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    # get coordiantes of yolov5 outputs
                    cx, cy, w, h = detect[0], detect[1], detect[2], detect[3]
                    print(cx, cy, w, h)

                    # calculate 
                    left= int((cx - w/2)* x_scale)

                    top= int((cy - h/2)* y_scale)

                    width = int(w * x_scale)

                    height = int(h *y_scale)

                    box= np.array([left, top, width, height])

                    bboxes.append(box)

        #np.array(score)
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold)
        # print(indices)
        for i in indices:

            box = bboxes[i] #box = bboxes[i[0]]

            left = box[0]

            top = box[1]

            width = box[2]
            
            height = box[3] 
           
           # cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 3)
            
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])

      
            cv2.rectangle(img, (left, top),(left + width, top + height), (0,255,0),3)
            cv2.putText(img, label, (left, top+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
           
        cv2.namedWindow('result.jpg', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result.jpg', 900, 800)
        cv2.imshow('result.jpg', img)
            # cv2.imshow('output',img)    
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        

'''
 if __name__ == "__main__":
   parser = argparse.ArgumentParser()
    parser.add_argument('--image',help='Specify input image', default= '', type=str)
    parser.add_argument('--weights', default='./yolov5s.onnx', type=str, help='model weights path')
    parser.add_argument('--imgs_w', default=640,type= int, help='image size')
    parser.add_argument('--imgs_h', default=640,type= int, help='image size')
    parser.add_argument('--conf_thres',default= 0.7, type=float, help='confidence threshold')
    parser.add_argument('--score_thres',type= float, default= 0.5, help='iou threshold')
    parser.add_argument('--nms_thres',type= float, default= 0.5, help='nms threshold')
    parser.add_argument('--classes',type=str,default='', help='class names')
    opt= parser.parse_args()
    instance= detectv5( opt.image, opt.weights, opt.imgs_w, opt.imgs_h, opt.conf_thres, opt.score_thres, opt.nms_thres, opt.classes)
    instance()'''
def main():
    image_path = '/tensorfl_vision/Onnx-Inference-Yolov5/Images'
    onnx_path = '/tensorfl_vision/Onnx-Inference-Yolov5/models/yolov5s.onnx'
    classes_path = '/tensorfl_vision/Onnx-Inference-Yolov5/coco-classes.txt'

    instance= detectv5(image_path, onnx_path, 640, 640, 0.34, 0.38, 0.3, classes_path)
    instance()


if __name__== "__main__":
    main()