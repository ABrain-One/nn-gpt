import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.81, 0.92), ratio=(0.99, 2.81)), transforms.RandomAdjustSharpness(sharpness_factor=0.82, p=0.19), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])