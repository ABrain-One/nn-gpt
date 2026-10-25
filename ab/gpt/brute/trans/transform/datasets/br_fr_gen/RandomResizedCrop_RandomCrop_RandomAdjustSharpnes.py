import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.93), ratio=(1.22, 2.89)),
    transforms.RandomCrop(size=28),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
