import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 1.0), ratio=(1.0, 1.25)), transforms.RandomAdjustSharpness(sharpness_factor=1.28, p=0.77), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])