import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.81), ratio=(0.84, 2.2)),
    transforms.ColorJitter(brightness=0.96, contrast=1.04, saturation=1.15, hue=0.08),
    transforms.RandomCrop(size=30),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
