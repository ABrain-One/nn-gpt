import torch
import torchvision.transforms as F

def transform(norm):
    return F.Compose([F.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.8, 1.2)), F.RandomHorizontalFlip(p=0.5), F.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), F.RandomApply([F.GaussianBlur(kernel_size=3)], p=0.5), F.ToTensor(), F.Normalize(**norm)])