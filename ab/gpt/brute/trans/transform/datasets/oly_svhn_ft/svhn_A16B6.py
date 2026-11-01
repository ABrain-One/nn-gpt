import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.4), transforms.RandomInvert(p=0.13), transforms.RandomPosterize(bits=8, p=0.58), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])