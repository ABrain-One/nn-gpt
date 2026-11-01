import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomHorizontalFlip(p=0.14),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.9), ratio=(1.17, 2.19)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
