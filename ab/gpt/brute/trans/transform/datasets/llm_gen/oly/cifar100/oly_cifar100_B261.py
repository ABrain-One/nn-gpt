import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.RandomCrop(25), transforms.RandomAutocontrast(p=0.8), transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.6), transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])