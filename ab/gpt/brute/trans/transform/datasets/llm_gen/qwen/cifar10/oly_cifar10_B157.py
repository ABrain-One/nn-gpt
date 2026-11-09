import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.75, 1.25)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.5, 1.5)), transforms.RandomPerspective(distortion_scale=0.2, p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])