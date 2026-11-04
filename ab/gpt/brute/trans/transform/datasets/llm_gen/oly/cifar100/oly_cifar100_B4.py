import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.3)), transforms.RandomPerspective(distortion_scale=0.25, p=0.5), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])