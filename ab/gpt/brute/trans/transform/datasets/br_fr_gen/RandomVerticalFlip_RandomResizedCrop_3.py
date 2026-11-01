import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.32),
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.97), ratio=(0.95, 2.18)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
