import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.67, 0.99), ratio=(1.2, 1.58)), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomApply([transforms.GaussianBlur(3, sigma=1.5)], p=0.5), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])