import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None,size_average=True):
    
        super (BCEDiceLoss,self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        
        #binary segmentation
        inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
if __name__ == "__main__":
    inputs = torch.tensor([[0,1,1,1]])
    targets = torch.tensor([[0,1,1,1]])
    
    dice_bce = BCEDiceLoss()
    print(dice_bce.forward(inputs, targets))
    