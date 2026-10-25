import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.08, contrast=1.04, saturation=0.98, hue=0.05),
    transforms.RandomRotation(degrees=11),
    transforms.RandomAutocontrast(p=0.85),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
