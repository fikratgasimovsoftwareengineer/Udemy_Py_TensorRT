import torch
import torch.nn as nn

class AtrousConv(nn.Module):
    def __init__(self, input_channels, kernel_size, pad, dilation, output_channels=256):
        super(AtrousConv, self).__init__()
    
       
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, 
                              kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x