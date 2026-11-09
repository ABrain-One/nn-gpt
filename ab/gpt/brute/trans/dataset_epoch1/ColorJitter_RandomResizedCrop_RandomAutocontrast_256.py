import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.84, contrast=1.18, saturation=1.15, hue=0.09),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 1.0), ratio=(1.11, 1.71)),
    transforms.RandomAutocontrast(p=0.62),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
