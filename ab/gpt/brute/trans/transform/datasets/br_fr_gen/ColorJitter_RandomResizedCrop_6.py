import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.02, contrast=1.2, saturation=1.12, hue=0.04),
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.9), ratio=(1.21, 2.54)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
