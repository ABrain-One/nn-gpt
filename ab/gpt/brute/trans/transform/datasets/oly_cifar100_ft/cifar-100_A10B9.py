import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 0.95), ratio=(0.89, 1.68)), transforms.RandomAdjustSharpness(sharpness_factor=1.12, p=0.18), transforms.GaussianBlur(kernel_size=5, sigma=(0.88, 1.57)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])