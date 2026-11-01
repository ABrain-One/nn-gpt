import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 2.0), ratio=(0.8, 1.3333333333333333)), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), transforms.RandomAffine(degrees=30, translate=(0.3, 0.2), scale=(0.8, 1.2), shear=(5, 10), fill='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])