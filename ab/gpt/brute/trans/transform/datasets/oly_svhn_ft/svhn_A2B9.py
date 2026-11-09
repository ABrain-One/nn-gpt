import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.73, 0.9), ratio=(0.87, 1.44)), transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=0.25), transforms.RandomPosterize(bits=7, p=0.69), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])