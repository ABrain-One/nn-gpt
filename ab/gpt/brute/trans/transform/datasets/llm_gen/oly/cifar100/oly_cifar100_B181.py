import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomInvert(p=0.9), transforms.GaussianBlur(kernel_size=5, sigma=(0.78, 1.03)), transforms.RandomRotation(degrees=5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])