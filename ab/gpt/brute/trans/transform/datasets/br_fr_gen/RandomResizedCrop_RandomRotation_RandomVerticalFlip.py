import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.82), ratio=(1.17, 2.27)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomVerticalFlip(p=0.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
