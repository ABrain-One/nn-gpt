import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.84, contrast=0.94, saturation=0.82, hue=0.08),
    transforms.RandomRotation(degrees=7),
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.92), ratio=(1.25, 1.87)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
