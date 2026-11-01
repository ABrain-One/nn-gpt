import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.86), ratio=(1.08, 2.2)),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
