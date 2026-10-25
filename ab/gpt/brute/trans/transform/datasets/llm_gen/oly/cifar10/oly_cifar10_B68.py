import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.9), transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), transforms.RandomPerspective(distortion_scale=0.28, p=0.66), transforms.ToTensor(), transforms.Normalize(*norm)])