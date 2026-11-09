import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.64),
    transforms.RandomAdjustSharpness(sharpness_factor=1.39, p=0.38),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.85), ratio=(0.86, 2.24)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
