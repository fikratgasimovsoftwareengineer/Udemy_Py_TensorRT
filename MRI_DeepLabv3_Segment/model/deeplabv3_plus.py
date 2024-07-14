from torchvision.models import resnet101
import torch
import torch.nn as nn
import sys
from pathlib import Path
base_dir = Path('/all_vision/MRI_DeepLabv3_Segment')

sys.path.append(str(base_dir))

from model.deeplabv3_aspp import ASPP

import torchvision.models as models

class DeepLabV3(nn.Module):
    def __init__(self, num_classes, backbone='resnet101'):
        
        super(DeepLabV3, self).__init__()
        
        if backbone=='resnet101':
        
            self.backbone = models.resnet101(pretrained=True)
            self.low_level_features = nn.Sequential(
                
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1
            )
            self.high_level_features = nn.Sequential(
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            )
            
            self.aspp = ASPP(in_channels=2048, out_channels=256)
            
            self.decoder = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

    def forward(self, x):
        
        low_level_feat = self.low_level_features(x)
        high_level_feat = self.high_level_features(low_level_feat)
        aspp_output = self.aspp(high_level_feat)
        
        aspp_output = nn.functional.interpolate(aspp_output, size=low_level_feat.size()[2:],
                                                mode='bilinear', align_corners=True)
        x = torch.cat([aspp_output, low_level_feat], dim=1)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)

        return x



