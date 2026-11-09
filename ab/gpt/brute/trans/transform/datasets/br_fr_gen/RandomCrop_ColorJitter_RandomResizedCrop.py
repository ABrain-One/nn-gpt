import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.ColorJitter(brightness=1.0, contrast=0.87, saturation=0.8, hue=0.06),
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.85), ratio=(1.18, 2.81)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
