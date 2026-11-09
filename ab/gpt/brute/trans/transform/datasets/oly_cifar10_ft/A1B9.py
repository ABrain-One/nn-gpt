import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.84), ratio=(1.12, 2.6)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.48, p=0.55),
    transforms.RandomHorizontalFlip(p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])