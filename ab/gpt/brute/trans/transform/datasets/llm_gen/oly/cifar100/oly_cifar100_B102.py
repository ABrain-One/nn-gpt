import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.25)), transforms.ColorJitter(0.4, 0.4, 0.4, 0.2), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(**norm)])