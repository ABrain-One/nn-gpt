import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.97, 2.02)), transforms.RandomHorizontalFlip(p=0.88), transforms.GaussianBlur(kernel_size=5, sigma=(0.95, 1.24)), transforms.RandomRotation(degrees=(-14, 14)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994])])