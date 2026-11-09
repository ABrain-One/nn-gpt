import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.79),
    transforms.ColorJitter(brightness=1.01, contrast=0.9, saturation=1.02, hue=0.09),
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.96), ratio=(0.94, 2.87)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
