import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.93, contrast=0.85, saturation=1.11, hue=0.07),
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.RandomAdjustSharpness(sharpness_factor=1.07, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
