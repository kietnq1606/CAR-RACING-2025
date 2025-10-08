import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def transform_to_tensor(CV2_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_transform = A.Compose([
        A.Resize(192, 320),
        A.Normalize(),
        ToTensorV2(),
    ])
    # BGR -> RGB
    CV2_image = cv2.cvtColor(CV2_image, cv2.COLOR_BGR2RGB)
    image = val_transform(image=CV2_image)["image"]
    image = image.unsqueeze(0).to(device)
    return image



