import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomAffine(degrees=13, translate=(0.11, 0.13), scale=(1.02, 1.67), shear=(2.46, 8.16)),
    transforms.RandomRotation(degrees=8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
