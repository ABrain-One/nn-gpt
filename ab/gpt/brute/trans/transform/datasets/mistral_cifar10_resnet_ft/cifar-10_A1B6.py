import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.88), transforms.RandomAdjustSharpness(sharpness_factor=1.23, p=0.19), transforms.RandomPosterize(bits=7, p=0.53), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])