import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.63), transforms.RandomPosterize(bits=4, p=0.14), transforms.RandomRotation(degrees=10), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])