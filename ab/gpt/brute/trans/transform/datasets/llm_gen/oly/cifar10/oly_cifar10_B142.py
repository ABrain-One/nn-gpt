import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)), transforms.RandomHorizontalFlip(), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05), transforms.ToTensor(), transforms.Normalize(mean=norm[0], std=norm[1])])