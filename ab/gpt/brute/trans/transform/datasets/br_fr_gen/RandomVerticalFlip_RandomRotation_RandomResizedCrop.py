import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.69),
    transforms.RandomRotation(degrees=21),
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.92), ratio=(0.8, 2.93)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
