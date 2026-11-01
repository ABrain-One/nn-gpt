import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.79, 0.85), ratio=(1.09, 1.74)), transforms.RandomPosterize(bits=8, p=0.2), transforms.RandomHorizontalFlip(p=0.48), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])