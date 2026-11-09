import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.99), ratio=(0.81, 1.66)),
    transforms.RandomInvert(p=0.8),
    transforms.RandomHorizontalFlip(p=0.28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
