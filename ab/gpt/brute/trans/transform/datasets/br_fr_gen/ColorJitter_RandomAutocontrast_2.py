import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.87, contrast=1.15, saturation=0.83, hue=0.08),
    transforms.RandomAutocontrast(p=0.13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
