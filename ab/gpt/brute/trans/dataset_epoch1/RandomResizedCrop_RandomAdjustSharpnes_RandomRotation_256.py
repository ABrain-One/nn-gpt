import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.92), ratio=(1.32, 1.68)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.86, p=0.87),
    transforms.RandomRotation(degrees=6),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
