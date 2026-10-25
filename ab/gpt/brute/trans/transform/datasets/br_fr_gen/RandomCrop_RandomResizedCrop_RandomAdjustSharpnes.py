import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.93), ratio=(1.2, 2.77)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.42, p=0.11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
