import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomVerticalFlip(), transforms.GaussianBlur(3, sigma=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])