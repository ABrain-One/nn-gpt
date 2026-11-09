import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.53), transforms.RandomPosterize(bits=5, p=0.85), transforms.RandomRotation(degrees=8), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])