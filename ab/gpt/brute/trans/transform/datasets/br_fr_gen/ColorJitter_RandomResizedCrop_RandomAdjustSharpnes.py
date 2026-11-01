import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=0.89, saturation=1.03, hue=0.01),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.86), ratio=(0.91, 2.62)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.86, p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
