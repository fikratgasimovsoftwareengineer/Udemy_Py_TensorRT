import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from torch import optim
from torchvision import transforms

import sys
from pathlib import Path

base_dir = Path('/all_vision/MRI_DeepLabv3_Segment')
sys.path.append(str(base_dir))

from model.dice_loss import DiceLoss
from model.deeplabv3_plus import DeepLabV3
from handler.dataloader_segment import LoadSegmentBrain


class TrainSegmentBrainMRI:
    
    def __init__(self, model, criterion, optimizer, train_loader,val_loader, num_epochs=10):
        self.model = model
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.__train_loader = train_loader
        self.__val_loader = val_loader
        self.num_epochs=num_epochs
        self.models_saved= base_dir / 'data/output'
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        self.model.to(self.device)
        
    def train_model(self, epoch):
        
        self.model.train()
        
        total_loss = 0
        total_predictions = 0
        total_correct = 0
        
        # loop throught dataloader train
        for batch_idx, (rgb_image, mask_image) in enumerate(self.__train_loader):
            
            rgb_image, mask_image = rgb_image.to(self.device), mask_image.to(self.device)
            
            self.optimizer.zero_grad()
            
            # model predictions
            outputs = self.model(rgb_image)
            
            loss = self.criterion(outputs, mask_image)
            
            # calculate params for loss
            total_loss += loss.item()
            loss.backward()
            
            self.optimizer.step()
            
            # calculate params for accuracy
            predictions = outputs > 0.5
            total_predictions += predictions.numel()
            total_correct += (predictions==mask_image).sum().item()
            
        accuracy = total_correct / total_predictions
        avg_loss = total_loss / len(self.__train_loader)
        print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss : {avg_loss:.4f},Train Accuracy {accuracy: .4f}")
        
        return avg_loss, accuracy
            
    def val_model(self, epoch):
        
        self.model.eval()
        total_loss = 0
        total_predictions= 0
        total_correct = 0
        
        with torch.inference_mode():
            for batch_idx, (rgb_images, mask_images) in enumerate(self.__val_loader):
                rgb_images, mask_images = rgb_images.to(self.device), mask_images.to(self.device)
                
                outputs = self.model(rgb_images)
                
                loss = self.criterion(outputs,mask_images)
                
                total_loss += loss.item()
                
                # accuracy
                predictions = outputs > 0.5
                total_predictions+= predictions.numel()
                total_correct += (predictions == mask_images).sum().item()
                
        accuracy = total_correct / total_predictions
        avg_loss = total_loss / len(self.__val_loader)
        
        print(f"Epoch [{epoch + 1}/{self.num_epochs}], Val Loss : {avg_loss:.4f}, Train Accuracy {accuracy: .4f}")
        return avg_loss, accuracy
    
    # syronize those steps: train & val
    def train_and_validate(self):
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy  = self.train_model(epoch)
            val_loss, val_accuracy = self.val_model(epoch)
            
            print("==================================")
            
            #update learning rate
            self.scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.models_saved}/best_model_deeplabv3_resnet101.pth')
            
        torch.save(self.model.state_dict(), f'{self.models_saved}/final_deeplabv3_resnet101.pth')
        print('Training DeepLabv3 Plus Complete..')


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
    
    # Instantiate Deeplabv3 plus
    num_classes =1
    model = DeepLabV3(num_classes)
    #=============================================
    #=########DICE LOSS ####################
    loss = DiceLoss()
    
    #=============Optimizer======================
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    
    trainSegmentMRIBrain = TrainSegmentBrainMRI(model, loss, optimizer, train_loader, val_loader)
    trainSegmentMRIBrain.train_and_validate()
    
if __name__ == "__main__":
    main()
    