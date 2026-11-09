import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.76, 0.98), ratio=(1.26, 2.61)), transforms.RandomHorizontalFlip(p=0.53), transforms.RandomAdjustSharpness(sharpness_factor=1.41, p=0.9), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])