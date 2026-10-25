import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.333)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.3), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.5), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(**norm)])