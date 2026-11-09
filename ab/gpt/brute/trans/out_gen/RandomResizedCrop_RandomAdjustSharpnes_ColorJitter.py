import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.98), ratio=(0.78, 2.62)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.7, p=0.33),
    transforms.ColorJitter(brightness=1.06, contrast=0.87, saturation=1.2, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
