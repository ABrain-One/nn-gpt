import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.73), transforms.RandomPosterize(bits=4, p=0.39), transforms.RandomAdjustSharpness(sharpness_factor=1.47, p=0.25), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])