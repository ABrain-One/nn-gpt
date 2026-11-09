import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.81), ratio=(0.95, 1.8)),
    transforms.RandomAffine(degrees=20, translate=(0.18, 0.12), scale=(1.17, 1.49), shear=(4.61, 7.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
