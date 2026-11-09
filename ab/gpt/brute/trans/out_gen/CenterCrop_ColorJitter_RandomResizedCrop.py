import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.ColorJitter(brightness=1.12, contrast=1.15, saturation=1.1, hue=0.07),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.92), ratio=(1.27, 2.36)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
