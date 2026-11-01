import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.89), ratio=(1.09, 2.15)),
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.RandomAdjustSharpness(sharpness_factor=1.12, p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])