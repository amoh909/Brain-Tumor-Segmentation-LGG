import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LGGSegmentationDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        """
            Args:
            csv_file (str): Path to CSV file containing patient_id, image_path, mask_path
            transform: Albumentations transform pipeline
        """    
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        ## Get file path from csv
        image_path = self.data_info.loc[idx, 'image_path']
        mask_path = self.data_info.loc[idx, 'mask_path']

        image_path = image_path.replace("../", "")
        mask_path = mask_path.replace("../", "")

        # Load image and mask in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found or could not be loaded: {mask_path}")
        
        if self.transform is not None:
            augmented = self.transform(image = image, mask = mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Normalization to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Converting the mask from [0, 255] to [0, 1], we converted to guarantee a binary mask.
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension: (H, W) -> (1, H, W) since PyTorch expects data of the latter form, the first form is how OpenCV loads images
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert from NumPy arrays to tensors for PyTorch
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask