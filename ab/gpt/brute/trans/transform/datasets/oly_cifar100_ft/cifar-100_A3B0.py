import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.11), transforms.RandomAutocontrast(p=0.78), transforms.GaussianBlur(kernel_size=3, sigma=(0.82, 1.54)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])