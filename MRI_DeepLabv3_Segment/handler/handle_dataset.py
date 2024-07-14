import pandas as pd
import os

import sys
from pathlib import Path

base_dir = Path('/all_vision/MRI_DeepLabv3_Segment')
sys.path.append(str(base_dir))

class DatasetHandle:
    
    def __init__(self, image_data):
        
        self.image_data = image_data
        
        self.csv_files_directory = base_dir / 'data/output'
        
    def read_images(self):
        dataset_collect = []
        for root,dirs,files in os.walk(self.image_data):
            for patient_dir in dirs:
                image_dir = os.path.join(root, patient_dir)

                # rgb and mask dictionary
                rgb_files = {}
                mask_files = {}
                
                for files in os.listdir(image_dir):
                    
                    if files.endswith('.tif') and not files.endswith('_mask.tif'):
                        basename = files.replace('.tif', '')
                        rgb_files[basename] = os.path.join(image_dir, files)
                        
                    elif files.endswith('_mask.tif'):
                        basename = files.replace('_mask.tif', '')
                        mask_files[basename] = os.path.join(image_dir, files)
                        
                for base_name in rgb_files.keys() and mask_files.keys():
                    dataset_collect.append({
                        "images_root":image_dir,
                        "rgb_image":rgb_files[base_name],
                        "mask_image":mask_files[base_name]
                        
                    })
                    
        dataset_dif = pd.DataFrame(dataset_collect)
        
        dataset_dif.to_csv(f"{self.csv_files_directory}/dataset_mri.csv", index=False)
        
def main():
    
    
    image_data = base_dir / 'data/input/lgg-mri-segmentation/kaggle_3m'
        
    datasetHandle = DatasetHandle(image_data=image_data)
    datasetHandle.read_images()
    
if __name__ == "__main__":
    main()
                    
                
                
            