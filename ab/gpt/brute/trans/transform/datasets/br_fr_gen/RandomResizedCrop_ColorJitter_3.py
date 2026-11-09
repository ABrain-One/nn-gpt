import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.84), ratio=(0.85, 2.43)),
    transforms.ColorJitter(brightness=0.88, contrast=1.13, saturation=1.07, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
