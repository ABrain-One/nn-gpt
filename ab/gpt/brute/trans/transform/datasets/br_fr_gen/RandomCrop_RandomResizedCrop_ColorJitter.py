import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.87), ratio=(0.76, 1.42)),
    transforms.ColorJitter(brightness=0.96, contrast=0.9, saturation=0.96, hue=0.1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
