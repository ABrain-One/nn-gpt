import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomPosterize(4, p=0.3), transforms.RandomPerspective(distortion_scale=0.2, p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(**norm)])