import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.78, 0.84), ratio=(1.02, 1.43)), transforms.RandomInvert(p=0.13), transforms.RandomPosterize(bits=7, p=0.53), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])