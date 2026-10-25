import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.81), ratio=(1.15, 1.53)),
    transforms.CenterCrop(size=26),
    transforms.RandomAffine(degrees=7, translate=(0.14, 0.11), scale=(1.11, 1.34), shear=(1.24, 8.97)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
