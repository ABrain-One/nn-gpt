import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.7), transforms.RandomAutocontrast(p=0.75), transforms.RandomPosterize(bits=8, p=0.19), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])