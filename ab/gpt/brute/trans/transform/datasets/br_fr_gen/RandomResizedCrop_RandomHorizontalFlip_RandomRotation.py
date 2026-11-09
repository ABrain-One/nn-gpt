import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.94), ratio=(0.96, 2.72)),
    transforms.RandomHorizontalFlip(p=0.73),
    transforms.RandomRotation(degrees=13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
