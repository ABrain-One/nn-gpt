import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.68, 0.95), ratio=(0.81, 1.68)), transforms.RandomPosterize(bits=7, p=0.16), transforms.RandomAdjustSharpness(sharpness_factor=1.73, p=0.57), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])