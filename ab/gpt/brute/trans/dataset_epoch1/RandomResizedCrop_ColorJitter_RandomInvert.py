import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.87), ratio=(1.14, 1.61)),
    transforms.ColorJitter(brightness=1.17, contrast=1.13, saturation=1.15, hue=0.08),
    transforms.RandomInvert(p=0.74),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
