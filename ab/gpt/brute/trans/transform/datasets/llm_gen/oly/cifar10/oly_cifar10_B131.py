import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(30), transforms.RandomPerspective(distortion_scale=0.12, p=0.5), transforms.RandomResizedCrop(28, scale=(0.8, 1.2)), transforms.Resize(64), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])