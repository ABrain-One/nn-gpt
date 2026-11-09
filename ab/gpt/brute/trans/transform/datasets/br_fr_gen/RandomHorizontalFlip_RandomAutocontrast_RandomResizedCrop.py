import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.41),
    transforms.RandomAutocontrast(p=0.85),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.81), ratio=(0.86, 1.9)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
