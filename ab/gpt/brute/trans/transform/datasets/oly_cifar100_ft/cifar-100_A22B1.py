import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.73, 0.95), ratio=(1.12, 2.58)), transforms.RandomPosterize(bits=7, p=0.65), transforms.RandomRotation(degrees=15), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])