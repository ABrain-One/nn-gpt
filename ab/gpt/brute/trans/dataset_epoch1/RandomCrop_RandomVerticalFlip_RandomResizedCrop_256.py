import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomVerticalFlip(p=0.31),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.91), ratio=(0.85, 1.88)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
