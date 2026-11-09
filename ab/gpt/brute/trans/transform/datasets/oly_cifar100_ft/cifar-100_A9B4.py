import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.77, 1.0), ratio=(1.02, 1.53)), transforms.RandomPosterize(bits=6, p=0.86), transforms.RandomCrop(size=28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])