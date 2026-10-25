import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.12, contrast=1.14, saturation=1.02, hue=0.06),
    transforms.RandomAffine(degrees=16, translate=(0.06, 0.02), scale=(0.85, 1.25), shear=(2.16, 8.36)),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.81), ratio=(1.3, 1.37)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
