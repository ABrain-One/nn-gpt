import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)), transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.2)), transforms.ToTensor(), transforms.Normalize(*norm)])