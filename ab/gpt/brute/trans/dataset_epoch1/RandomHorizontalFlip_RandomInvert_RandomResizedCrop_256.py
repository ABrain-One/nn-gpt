import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.59),
    transforms.RandomInvert(p=0.57),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(0.8, 1.79)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
