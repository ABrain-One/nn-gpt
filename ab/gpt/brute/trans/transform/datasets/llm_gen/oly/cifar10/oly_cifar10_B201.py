import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.7), transforms.RandomVerticalFlip(p=0.7), transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.333)), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), shear=5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201])])