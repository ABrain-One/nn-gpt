import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.5, 1.5), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=(-0.1, 0.1)), transforms.RandomVerticalFlip(p=0.05), transforms.GaussianBlur(kernel_size=3, sigma=(0.96, 1.7)), transforms.ToTensor(), transforms.Normalize(mean=norm[0], std=norm[1])])