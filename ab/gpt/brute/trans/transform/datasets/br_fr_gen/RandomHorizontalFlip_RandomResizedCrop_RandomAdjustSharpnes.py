import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.74),
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.95), ratio=(1.21, 1.41)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.01, p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
