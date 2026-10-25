import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=11),
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.83), ratio=(1.13, 1.94)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
