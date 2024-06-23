import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        
        super(FocalLoss, self).__init__()
        
    def forward(self, inputs, targets,alpha=None,gamma=None):
        
        inputs = F.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        BCE_Exp = torch.exp(-BCE)
        
        focal_loss = alpha * (1-BCE_Exp)**gamma * BCE
        
        return focal_loss