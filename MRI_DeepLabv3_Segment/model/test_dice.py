from pathlib import Path
import sys
import torch
base_dir = Path('/all_vision/Brain_Tumor_Segmentation')
sys.path.append(base_dir)

from dice_loss import DiceLoss

if __name__ == "__main__":
    inputs = torch.tensor([[0,1,1,1]])
    targets = torch.tensor([[0,1,1,1]])
    
    dice = DiceLoss()
    print(dice.forward(inputs, targets))