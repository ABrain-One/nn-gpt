import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.92), ratio=(0.96, 1.66)),
    transforms.ColorJitter(brightness=1.2, contrast=1.02, saturation=1.05, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
