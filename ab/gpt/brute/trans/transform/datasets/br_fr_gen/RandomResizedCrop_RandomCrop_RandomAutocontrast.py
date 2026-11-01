import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.99), ratio=(0.95, 1.76)),
    transforms.RandomCrop(size=30),
    transforms.RandomAutocontrast(p=0.13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
