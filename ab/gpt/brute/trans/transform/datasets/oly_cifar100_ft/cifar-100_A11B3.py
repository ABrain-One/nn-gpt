import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(1.31, 1.36)), transforms.GaussianBlur(kernel_size=3, sigma=(0.88, 1.37)), transforms.RandomEqualize(p=0.15), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])