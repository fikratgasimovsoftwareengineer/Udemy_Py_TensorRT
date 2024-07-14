import csv
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from PIL import Image


import sys
from pathlib import Path

base_dir = Path('/all_vision/MRI_DeepLabv3_Segment')
sys.path.append(str(base_dir))

class LoadSegmentBrain(Dataset):
    
    def __init__(self, dataframe, train=True, transform=None, transform_mask=None):
        super().__init__()
        self.__transform = transform
        self.__dataframe = dataframe
        self.__transform_mask = transform_mask
        
        self.final_data = None
        self.image_mask_pairs = np.array(self.handle_dataframe())
        
        train_pairs, val_pairs = train_test_split(self.image_mask_pairs, test_size=0.25, random_state=42)
        
        if train:
            self.final_data = train_pairs
        else:
            self.final_data = val_pairs
        

    # Return List
    def handle_dataframe(self):
        
        pair_image_mask = []
        
        with open(self.__dataframe, newline='') as data_segment:
            reader = csv.DictReader(data_segment)
            for row in reader:
                pair_image_mask.append([row["rgb_image"], row["mask_image"]])
            
        return pair_image_mask
    
    def __len__(self):
        return len(self.final_data)
    
    def __getitem__(self, idx_img):
        img_path, mask_path = self.final_data[idx_img]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.__transform:
            image = self.__transform(image)
        if self.__transform_mask:
            mask = self.__transform_mask(mask)
            
        return image, mask
    
    
    def display_images(self, save_dir, rgb_images, mask_images):
        rgb_images = rgb_images.numpy().transpose((0,2,3,1)) #[bath_size, channels, 256, 256]
        mask_images = mask_images.numpy().transpose((0,2,3,1))
        
        # normalized on rgb images
        mean = np.array([0.485, 0.456, 0.406])
        std =np.array([0.229, 0.224, 0.225])
        rgb_images = std * rgb_images + mean
        rgb_images = np.clip(rgb_images, 0, 1)
        
        fig, axes = plt.subplots(len(rgb_images), 2, figsize=(9, len(rgb_images) * 4))
        
        for i in range(len(rgb_images)):
            ax_image = axes[i, 0]
            ax_mask = axes[i,1]
            ax_image.imshow(rgb_images[i])
            ax_image.axis('off')
            ax_mask.imshow(mask_images[i].squeeze(), cmap='gray')
            ax_mask.axis('off')
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(os.path.join(save_dir, 'batch.png'))
        plt.close()
        
    
def main():
    dataframe = base_dir / 'data/output/dataset_mri.csv'
    
    transform = transforms.Compose([
        # Random Horizontally Flip
        transforms.RandomHorizontalFlip(),
        # Random Verticaly Flip
        transforms.RandomVerticalFlip(),
        # Randomly Rotate
        transforms.RandomRotation(15),
        # Randomly Resioze
        transforms.Resize((256, 256)),
        # convert to tensor
        transforms.ToTensor(),
        # Normalize
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_tranform = transforms.Compose([
        # Resize
        transforms.Resize((256, 256)),
        # Convert to Tensor
        transforms.ToTensor(),
    ])
    
    train_dataset = LoadSegmentBrain(dataframe, train=True, transform=transform, transform_mask=mask_tranform) 
    val_dataset = LoadSegmentBrain(dataframe, train=False, transform=transform, transform_mask=mask_tranform)
    print(len(train_dataset))
    print(len(val_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    save_dir = base_dir / 'data/output/visualized'
    for batch_idx, (rgb_images, mask_images) in enumerate(train_loader):
        if batch_idx==0:
            train_dataset.display_images(save_dir, rgb_images, mask_images)
            break
     
if __name__ == "__main__":
    main()
    
        
        