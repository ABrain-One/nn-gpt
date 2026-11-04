import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.91), ratio=(1.07, 2.61)),
    transforms.RandomRotation(degrees=8),
    transforms.RandomAffine(degrees=6, translate=(0.13, 0.17), scale=(1.12, 1.34), shear=(2.77, 6.85)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
