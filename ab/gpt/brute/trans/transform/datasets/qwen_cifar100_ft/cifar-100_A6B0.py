import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=(0.13, 1.14)), transforms.RandomAdjustSharpness(sharpness_factor=1.75, p=0.53), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])