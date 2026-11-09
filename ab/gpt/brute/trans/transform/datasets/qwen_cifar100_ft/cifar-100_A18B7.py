import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=30), transforms.GaussianBlur(kernel_size=3, sigma=(0.85, 1.0)), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])