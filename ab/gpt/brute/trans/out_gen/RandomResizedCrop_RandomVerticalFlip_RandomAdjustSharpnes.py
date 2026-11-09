import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.95), ratio=(1.28, 2.68)),
    transforms.RandomVerticalFlip(p=0.54),
    transforms.RandomAdjustSharpness(sharpness_factor=1.23, p=0.9),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
