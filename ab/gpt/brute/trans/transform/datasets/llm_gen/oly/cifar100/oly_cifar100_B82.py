import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale_jrange=(0.8, 1.2), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.5)], p=0.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])