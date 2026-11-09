import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.92), ratio=(0.98, 2.05)),
    transforms.RandomAutocontrast(p=0.58),
    transforms.RandomAffine(degrees=14, translate=(0.01, 0.13), scale=(1.14, 1.82), shear=(1.57, 9.1)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
