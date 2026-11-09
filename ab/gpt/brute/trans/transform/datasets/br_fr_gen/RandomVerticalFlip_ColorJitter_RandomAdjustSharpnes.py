import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.85),
    transforms.ColorJitter(brightness=0.97, contrast=1.05, saturation=1.0, hue=0.06),
    transforms.RandomAdjustSharpness(sharpness_factor=1.27, p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
