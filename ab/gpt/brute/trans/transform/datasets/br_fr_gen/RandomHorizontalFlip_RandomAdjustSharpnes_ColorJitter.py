import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomAdjustSharpness(sharpness_factor=1.94, p=0.62),
    transforms.ColorJitter(brightness=0.95, contrast=0.82, saturation=1.15, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
