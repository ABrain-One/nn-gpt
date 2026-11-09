import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.Pad(padding=4, padding_mode='symmetric'), transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5), transforms.RandomRotation(degrees=(-5, 5)), transforms.CenterCrop(size=32), transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])