import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(0.8, 1.95)), transforms.RandomPosterize(bits=7, p=0.65), transforms.RandomCrop(size=32), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])