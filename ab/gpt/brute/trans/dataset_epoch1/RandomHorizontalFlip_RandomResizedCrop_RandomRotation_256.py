import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.61),
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.81), ratio=(0.89, 1.52)),
    transforms.RandomRotation(degrees=20),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
