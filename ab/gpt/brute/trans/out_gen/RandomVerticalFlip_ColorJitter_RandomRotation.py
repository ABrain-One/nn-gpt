import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.45),
    transforms.ColorJitter(brightness=1.13, contrast=1.14, saturation=0.81, hue=0.05),
    transforms.RandomRotation(degrees=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
