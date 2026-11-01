import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=22), transforms.RandomHorizontalFlip(p=0.3), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])