import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.46), transforms.RandomPosterize(bits=8, p=0.32), transforms.RandomAdjustSharpness(sharpness_factor=0.77, p=0.57), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])