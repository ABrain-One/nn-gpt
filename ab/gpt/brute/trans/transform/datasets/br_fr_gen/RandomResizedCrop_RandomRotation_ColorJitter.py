import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.97), ratio=(1.29, 2.36)),
    transforms.RandomRotation(degrees=24),
    transforms.ColorJitter(brightness=0.87, contrast=1.05, saturation=1.09, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
