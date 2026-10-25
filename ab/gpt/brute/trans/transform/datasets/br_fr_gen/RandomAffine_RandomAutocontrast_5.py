import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=16, translate=(0.16, 0.03), scale=(1.05, 1.41), shear=(3.15, 9.4)),
    transforms.RandomAutocontrast(p=0.24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
