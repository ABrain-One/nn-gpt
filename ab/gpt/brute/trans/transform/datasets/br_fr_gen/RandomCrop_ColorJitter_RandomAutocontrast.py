import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.ColorJitter(brightness=0.97, contrast=0.87, saturation=1.15, hue=0.02),
    transforms.RandomAutocontrast(p=0.41),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
