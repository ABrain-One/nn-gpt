import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(128, 128, 128), padding_mode='constant'), transforms.RandomAffine(degrees=15, translate=(0.1, 0.2), scale=(0.8, 1.0), shear=(-5.0, 5.0)), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.Resize(64), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])