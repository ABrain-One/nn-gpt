import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.35), transforms.RandomResizedCrop(size=32, scale=(0.8, 0.93), ratio=(1.05, 1.93)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])