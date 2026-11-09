import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.73, 0.8), ratio=(0.8, 1.41)), transforms.RandomPosterize(bits=4, p=0.46), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])