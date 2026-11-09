import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.87, contrast=1.08, saturation=1.17, hue=0.01),
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.85), ratio=(1.1, 2.7)),
    transforms.CenterCrop(size=30),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
