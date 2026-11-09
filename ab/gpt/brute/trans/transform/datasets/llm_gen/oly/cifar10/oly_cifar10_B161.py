import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.5), transforms.RandomRotation(degrees=(-5, 5)), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(40), transforms.RandomCrop(32), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(**norm)])