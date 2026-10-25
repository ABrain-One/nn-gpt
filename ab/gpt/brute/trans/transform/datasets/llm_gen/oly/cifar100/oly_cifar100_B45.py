import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.RandomApply([transforms.Autocontrast()], p=0.5), transforms.RandomResizedCrop(32, scale=(0.6, 0.9), ratio=(0.9, 1.1)), transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8)), transforms.RandomAffine(degrees=15, translate=(-0.2, 0.2), scale=(0.7, 1.2), shear=(-8.0, 8.0)), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])