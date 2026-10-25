import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(1.19, 2.35)),
    transforms.RandomVerticalFlip(p=0.21),
    transforms.RandomRotation(degrees=4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
