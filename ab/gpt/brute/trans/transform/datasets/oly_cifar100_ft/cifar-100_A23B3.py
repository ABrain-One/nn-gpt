import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=32), transforms.RandomAutocontrast(p=0.26), transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(1.15, 1.43)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])