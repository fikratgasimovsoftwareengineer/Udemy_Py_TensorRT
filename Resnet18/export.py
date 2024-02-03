'''
    Export Resnet 18 from torchvision Models
    and export as an onnx with image resolition : 1, 3, 224, 224

'''

import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)

resnet18_image = torch.rand(1, 3, 224, 224)

torch.onnx.export(model, resnet18_image,  "/tensorfl_vision/Resnet18/resnet18.onnx")