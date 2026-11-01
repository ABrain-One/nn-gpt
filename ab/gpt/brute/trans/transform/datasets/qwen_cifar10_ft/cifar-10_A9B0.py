import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=27), transforms.RandomAutocontrast(p=0.41), transforms.RandomResizedCrop(size=32, scale=(0.78, 0.88), ratio=(1.19, 1.41)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])