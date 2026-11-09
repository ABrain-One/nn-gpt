import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.Resize(64), transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.8, 1.25)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), transforms.RandomAdjustSharpness(sharpness_factor=0.75, p=0.5), transforms.Solarize(threshold=95, p=0.3), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])