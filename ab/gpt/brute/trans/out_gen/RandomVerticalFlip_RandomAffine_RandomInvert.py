import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.58),
    transforms.RandomAffine(degrees=14, translate=(0.17, 0.13), scale=(1.02, 1.55), shear=(4.42, 8.85)),
    transforms.RandomInvert(p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
