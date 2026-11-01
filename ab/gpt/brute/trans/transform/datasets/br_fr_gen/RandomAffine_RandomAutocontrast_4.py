import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=13, translate=(0.11, 0.13), scale=(0.87, 1.93), shear=(4.55, 6.79)),
    transforms.RandomAutocontrast(p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
