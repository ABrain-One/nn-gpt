import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.9, contrast=0.8, saturation=1.1, hue=0.1), transforms.RandomPerspective(distortion_scale=0.26, p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAdjustSharpness(1.07, p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])