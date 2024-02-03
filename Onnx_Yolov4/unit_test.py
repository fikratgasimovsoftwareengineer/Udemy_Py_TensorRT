
class_path = '/tensorfl_vision/Onnx_Yolov4/coco-classes.txt'

with open(class_path, 'r') as read_class:
    classes = [ class2.strip() for class2 in read_class.readlines()]
    print(classes)