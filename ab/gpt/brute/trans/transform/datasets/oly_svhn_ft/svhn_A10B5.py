import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=32), transforms.RandomAdjustSharpness(sharpness_factor=0.83, p=0.83), transforms.RandomPosterize(bits=7, p=0.17), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])