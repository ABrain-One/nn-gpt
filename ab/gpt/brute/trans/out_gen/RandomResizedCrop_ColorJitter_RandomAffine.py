import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.87), ratio=(1.32, 2.47)),
    transforms.ColorJitter(brightness=1.01, contrast=1.12, saturation=0.87, hue=0.08),
    transforms.RandomAffine(degrees=28, translate=(0.15, 0.0), scale=(1.15, 1.54), shear=(0.58, 8.53)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
