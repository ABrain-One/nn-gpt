import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8), transforms.RandomRotation(degrees=(-10, 10), p=0.8), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=1, sigma=0.5, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.2023, 0.2023])])