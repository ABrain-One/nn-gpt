import torch
import torchvision.transforms as F
from torchvision import transforms

def transform(norm):
    return transforms.Compose([F.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.9, 2.27)), F.ColorJitter(0.3, 0.3, 0.3, 0.1), F.RandomRotation(15), F.RandomHorizontalFlip(0.7), F.RandomVerticalFlip(0.3), F.GaussianBlur(kernel_size=3, sigma=1.5), F.ToTensor(), F.Normalize(**norm)])