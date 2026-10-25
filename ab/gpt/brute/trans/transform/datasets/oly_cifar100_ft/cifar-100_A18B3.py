import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPosterize(bits=7, p=0.15), transforms.RandomHorizontalFlip(p=0.73), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])