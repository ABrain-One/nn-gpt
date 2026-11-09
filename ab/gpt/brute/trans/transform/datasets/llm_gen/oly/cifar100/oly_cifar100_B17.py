import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])