import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)), transforms.RandomVerticalFlip(p=0.7), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.ToTensor(), transforms.Normalize(mean=[125.3 / 255, 123.1 / 255, 113.7 / 255], std=[63.0 / 255, 62.1 / 255, 66.75 / 255])])