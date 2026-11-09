import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.GaussianBlur(kernel_size=3, sigma=(0.58, 1.65)), transforms.RandomAutocontrast(p=0.68), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])