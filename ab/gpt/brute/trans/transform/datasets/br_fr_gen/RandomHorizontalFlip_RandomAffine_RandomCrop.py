import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.RandomAffine(degrees=23, translate=(0.05, 0.11), scale=(0.97, 1.39), shear=(1.39, 6.17)),
    transforms.RandomCrop(size=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
