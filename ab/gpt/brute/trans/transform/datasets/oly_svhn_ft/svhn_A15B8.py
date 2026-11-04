import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.8), transforms.RandomAdjustSharpness(sharpness_factor=1.37, p=0.22), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])