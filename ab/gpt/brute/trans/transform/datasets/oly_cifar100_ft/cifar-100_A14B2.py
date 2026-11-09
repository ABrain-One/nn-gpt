import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.13), transforms.RandomGrayscale(p=0.82), transforms.RandomResizedCrop(size=32, scale=(0.55, 0.93), ratio=(0.82, 2.45)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])