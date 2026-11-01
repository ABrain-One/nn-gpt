import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomAdjustSharpness(sharpness_factor=0.6, p=0.87), transforms.RandomPosterize(bits=7, p=0.11), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])