import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.73, 0.9), ratio=(1.21, 2.48)), transforms.RandomInvert(p=0.65), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])