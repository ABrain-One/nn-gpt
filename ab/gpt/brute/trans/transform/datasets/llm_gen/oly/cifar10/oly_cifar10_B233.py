import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.9), transforms.RandomInvert(p=0.9), transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.RandomRotation(degrees=10, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201])])