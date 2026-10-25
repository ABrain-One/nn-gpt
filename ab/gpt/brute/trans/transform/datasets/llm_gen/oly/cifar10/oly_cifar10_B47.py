import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(4, padding_mode='reflect'), transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.RandomAutocontrast(p=0.2), transforms.ToTensor(), transforms.Normalize(mean=[0.49139975, 0.48215856, 0.4465308], std=[0.2023362, 0.19946434, 0.20101056])])