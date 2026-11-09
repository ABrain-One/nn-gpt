import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.82), ratio=(0.83, 1.75)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.83, p=0.26),
    transforms.RandomHorizontalFlip(p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
