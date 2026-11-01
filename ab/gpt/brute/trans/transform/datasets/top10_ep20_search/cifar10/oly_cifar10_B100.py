import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.ToTensor(), transforms.Normalize(*norm)])