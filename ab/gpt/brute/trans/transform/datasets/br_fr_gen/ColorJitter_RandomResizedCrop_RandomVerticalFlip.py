import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.13, contrast=0.96, saturation=0.82, hue=0.05),
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.82), ratio=(1.18, 2.02)),
    transforms.RandomVerticalFlip(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
