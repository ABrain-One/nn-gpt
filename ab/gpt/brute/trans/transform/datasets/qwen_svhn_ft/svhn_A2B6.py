import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.GaussianBlur(kernel_size=3, sigma=(0.49, 1.41)), transforms.RandomAdjustSharpness(sharpness_factor=0.98, p=0.39), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])