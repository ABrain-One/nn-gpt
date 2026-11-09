import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.83),
    transforms.ColorJitter(brightness=1.03, contrast=1.13, saturation=1.05, hue=0.09),
    transforms.RandomAutocontrast(p=0.85),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
