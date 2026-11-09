import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.81),
    transforms.RandomAffine(degrees=20, translate=(0.14, 0.19), scale=(1.1, 1.27), shear=(2.55, 8.61)),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.81), ratio=(1.05, 1.75)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
