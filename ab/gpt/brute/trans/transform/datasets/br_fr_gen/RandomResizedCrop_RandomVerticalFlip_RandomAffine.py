import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.81), ratio=(0.82, 1.64)),
    transforms.RandomVerticalFlip(p=0.76),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.19), scale=(0.99, 1.31), shear=(1.05, 9.95)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
