import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.13), transforms.RandomPosterize(bits=5, p=0.83), transforms.RandomAdjustSharpness(sharpness_factor=1.55, p=0.78), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])