import sys
from pathlib import Path
base_dir = Path('/all_vision/MRI_DeepLabv3_Segment')
from model.deeplabv3_atconv import AtrousConv
import torch 
import torch.nn as nn


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        """Atrous Spatial Pyramid pooling layer
        Args:
            in_channles (int): No of input channel for Atrous_Convolution.
            out_channles (int): No of output channel for Atrous_Convolution.
        """
        super(ASPP, self).__init__()
        
        self.atrous_block1 = AtrousConv(input_channels=in_channels, output_channels=out_channels, kernel_size=1, pad=0, dilation=1)
        self.atrous_block6 = AtrousConv(input_channels=in_channels, output_channels=out_channels, kernel_size=3, pad=6, dilation=6)
        self.atrous_block12 = AtrousConv(input_channels=in_channels, output_channels=out_channels, kernel_size= 3, pad=12, dilation=12)
        self.atrous_block18 = AtrousConv(input_channels=in_channels, output_channels=out_channels,kernel_size= 3, pad=18, dilation=18)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,out_channels,
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        
        
        self.conv1_final = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)
        self.batchnorm_final = nn.BatchNorm2d(out_channels)
        self.relu_final = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x  = self.conv1_final(x)
        x = self.batchnorm_final(x)
        x = self.relu_final(x)
        return  x
