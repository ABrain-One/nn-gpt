import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(80), transforms.RandomCrop(64), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])