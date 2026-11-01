import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.5, 2.0)), transforms.RandomHorizontalFlip(p=0.3), transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])