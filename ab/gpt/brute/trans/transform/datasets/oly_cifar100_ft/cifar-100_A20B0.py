import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomAutocontrast(p=0.37), transforms.RandomResizedCrop(size=32, scale=(0.7, 0.83), ratio=(1.05, 2.75)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])