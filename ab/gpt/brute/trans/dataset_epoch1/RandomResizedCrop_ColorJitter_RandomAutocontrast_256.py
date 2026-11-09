import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.86), ratio=(0.94, 1.37)),
    transforms.ColorJitter(brightness=0.98, contrast=0.86, saturation=1.04, hue=0.01),
    transforms.RandomAutocontrast(p=0.31),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
