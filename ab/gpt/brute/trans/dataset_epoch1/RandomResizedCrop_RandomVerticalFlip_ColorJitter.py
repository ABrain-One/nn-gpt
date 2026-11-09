import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.93), ratio=(1.11, 1.59)),
    transforms.RandomVerticalFlip(p=0.84),
    transforms.ColorJitter(brightness=1.09, contrast=1.12, saturation=0.91, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
