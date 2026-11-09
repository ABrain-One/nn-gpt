import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.29),
    transforms.RandomAutocontrast(p=0.68),
    transforms.ColorJitter(brightness=1.13, contrast=1.11, saturation=1.14, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
