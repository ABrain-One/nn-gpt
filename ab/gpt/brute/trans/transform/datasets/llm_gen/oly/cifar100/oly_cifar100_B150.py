import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale_range=(0.5, 2.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomApply([transforms.RandomInvert()], p=0.2), transforms.RandomAffine(degrees=10, translate=(0.1, 0.2), scale=(0.8, 1.2), shear=5.0), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(**norm)])