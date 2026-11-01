import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.84), ratio=(1.0, 1.68)),
    transforms.RandomCrop(size=28),
    transforms.RandomVerticalFlip(p=0.28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
