import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8), transforms.RandomApply([transforms.GaussianBlur(3, sigma=1.5)], p=0.1), transforms.RandomRotation(degrees=(-15, 15), p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])