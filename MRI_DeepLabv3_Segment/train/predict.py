import torch
import numpy as np
import torch.nn as nn
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch import optim
import matplotlib.pyplot as plt
import os

base_dir = Path('/all_vision/MRI_DeepLabv3_Segment')
sys.path.append(str(base_dir))

from model.dice_loss import DiceLoss
from model.deeplabv3_plus import DeepLabV3
from handler.dataloader_segment import LoadSegmentBrain

class Inference:
    
    def predict(self, model: nn.Module, sample_loader: torch.utils.data.DataLoader, device: torch.device,threshold=0.6):
        model.to(device)
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.inference_mode():
            for batch_idx, (images, masks) in enumerate(sample_loader):
                images, masks = images.to(device), masks.to(device)
                y_pred = model(images)
                y_pred = torch.sigmoid(y_pred)
                y_pred = (y_pred > threshold).float()
                predictions.append(y_pred.cpu().numpy())
                true_labels.append(masks.cpu().numpy())

        return np.vstack(predictions), np.concatenate(true_labels)

    def save_images(self, save_dir, rgb_images, mask_images, pred_images):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for i in range(len(rgb_images)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Denormalize and transpose RGB images
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            rgb_image = rgb_images[i].numpy().transpose((1, 2, 0))
            rgb_image = std * rgb_image + mean
            rgb_image = np.clip(rgb_image, 0, 1)

            mask_image = mask_images[i].numpy().squeeze()
            pred_image = pred_images[i].squeeze()

            axes[0].imshow(rgb_image)
            axes[0].set_title('RGB Image')
            axes[0].axis('off')

            axes[1].imshow(mask_image, cmap='gray')
            axes[1].set_title('Mask Image')
            axes[1].axis('off')

            axes[2].imshow(pred_image, cmap='gray')
            axes[2].set_title('Predicted Image')
            axes[2].axis('off')

            plt.savefig(os.path.join(save_dir, f'image_{i}.png'))
            plt.close()

def main():
    dataframe = base_dir / 'data/output/dataset_mri.csv'
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Mask images should not be normalized
    transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = LoadSegmentBrain(dataframe, train=True, transform=transform, transform_mask=transform_mask)
    val_dataset = LoadSegmentBrain(dataframe, train=False, transform=transform, transform_mask=transform_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Instantiate the model
    num_classes = 1  # Binary segmentation: tumor vs. no tumor
    model = DeepLabV3(num_classes)
    model.load_state_dict(torch.load('/all_vision/Brain_Tumor_Segmentation/data/pretrained_models/final_deeplabv3_resnet101.pth'))  # Load the pre-trained model
    
    # Define inference class
    inference = Inference()

    # Perform inference on a batch of validation images
    for batch_idx, (images, masks) in enumerate(val_loader):
        
        predictions, true_labels = inference.predict(model, [(images, masks)], device)

    print('FINISHED!')

    # Save the images
    save_dir = './output_images'
    inference.save_images(save_dir, images, masks, predictions)

if __name__ == "__main__":
    main()
