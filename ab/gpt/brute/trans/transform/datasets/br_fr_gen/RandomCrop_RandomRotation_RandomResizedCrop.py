import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomRotation(degrees=27),
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.9), ratio=(0.92, 2.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
