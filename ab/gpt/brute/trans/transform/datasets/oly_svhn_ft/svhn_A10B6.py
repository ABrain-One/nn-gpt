import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(1.22, 2.88)), transforms.RandomInvert(p=0.28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])