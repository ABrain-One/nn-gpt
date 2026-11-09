import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.84), ratio=(1.01, 2.13)),
    transforms.CenterCrop(size=28),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
