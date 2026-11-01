import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.96, contrast=1.03, saturation=0.86, hue=0.04),
    transforms.RandomAdjustSharpness(sharpness_factor=1.13, p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
