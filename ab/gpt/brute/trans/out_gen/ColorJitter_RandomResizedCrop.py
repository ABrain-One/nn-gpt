import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.13, contrast=1.1, saturation=1.02, hue=0.05),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.95), ratio=(1.31, 2.33)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
