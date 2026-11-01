import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.89), ratio=(0.84, 2.04)),
    transforms.ColorJitter(brightness=0.81, contrast=0.96, saturation=1.15, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
