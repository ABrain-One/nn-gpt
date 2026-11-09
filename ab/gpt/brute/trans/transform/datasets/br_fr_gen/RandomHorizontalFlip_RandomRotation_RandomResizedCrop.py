import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.85),
    transforms.RandomRotation(degrees=16),
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.83), ratio=(1.26, 2.6)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
