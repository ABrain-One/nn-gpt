import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-10, 10)), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(*norm)])