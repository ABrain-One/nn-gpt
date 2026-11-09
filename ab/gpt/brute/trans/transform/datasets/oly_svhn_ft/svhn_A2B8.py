import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=(0.32, 1.55)), transforms.RandomAutocontrast(p=0.77), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])