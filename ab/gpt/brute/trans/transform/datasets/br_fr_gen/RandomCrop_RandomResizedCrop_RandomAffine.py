import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.84), ratio=(0.96, 2.02)),
    transforms.RandomAffine(degrees=25, translate=(0.03, 0.03), scale=(1.12, 1.65), shear=(0.13, 5.59)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
