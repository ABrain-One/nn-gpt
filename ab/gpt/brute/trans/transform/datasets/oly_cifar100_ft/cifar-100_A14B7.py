import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.RandomRotation(degrees=16), transforms.RandomPosterize(bits=7, p=0.51), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])