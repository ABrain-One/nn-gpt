import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.97), ratio=(1.13, 2.04)),
    transforms.RandomRotation(degrees=9),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
