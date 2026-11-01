import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.18, contrast=1.16, saturation=1.05, hue=0.08),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.93), ratio=(0.88, 1.42)),
    transforms.RandomCrop(size=31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
