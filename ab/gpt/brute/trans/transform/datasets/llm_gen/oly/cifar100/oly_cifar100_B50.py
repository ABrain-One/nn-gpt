import torch
import torchvision.transforms as F

def transform(norm):
    return F.Compose([F.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), F.RandomHorizontalFlip(p=0.5), F.ColorJitter(brightness=0.9, contrast=1.2, saturation=1.2, hue=0.1), F.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), F.Resize((64, 64)), F.ToTensor(), F.Normalize(*norm)])