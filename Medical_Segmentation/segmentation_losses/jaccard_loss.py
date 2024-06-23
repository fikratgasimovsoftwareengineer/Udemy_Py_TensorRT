import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # intersection
        '''
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        '''
        intersection = (inputs*targets).sum()
        # total
        total = (inputs + targets).sum()   
        # union 
        union = total - intersection
        
        IoU = (intersection + smooth) / (union + smooth) 
        return 1-IoU