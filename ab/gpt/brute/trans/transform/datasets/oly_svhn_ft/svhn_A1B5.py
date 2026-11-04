import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.RandomPosterize(bits=7, p=0.74), transforms.RandomAdjustSharpness(sharpness_factor=1.12, p=0.25), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])