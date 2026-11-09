import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.3)), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), transforms.ToTensor(), transforms.Normalize(**norm)])