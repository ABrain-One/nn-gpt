import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.9), ratio=(1.23, 2.38)),
    transforms.RandomHorizontalFlip(p=0.63),
    transforms.RandomAutocontrast(p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
