import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.45),
    transforms.ColorJitter(brightness=0.88, contrast=1.05, saturation=1.14, hue=0.04),
    transforms.RandomAutocontrast(p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
