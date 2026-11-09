import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.95), ratio=(1.12, 2.58)),
    transforms.RandomCrop(size=26),
    transforms.RandomHorizontalFlip(p=0.36),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
