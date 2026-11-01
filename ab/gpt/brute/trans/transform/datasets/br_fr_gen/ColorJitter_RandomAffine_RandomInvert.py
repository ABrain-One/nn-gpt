import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.05, contrast=1.0, saturation=0.82, hue=0.06),
    transforms.RandomAffine(degrees=15, translate=(0.19, 0.13), scale=(1.14, 1.41), shear=(4.58, 9.82)),
    transforms.RandomInvert(p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
