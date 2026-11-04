import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=0.53, p=0.41),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.9), ratio=(1.14, 1.86)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
