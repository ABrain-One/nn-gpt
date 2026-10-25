import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(15), transforms.Resize(256), transforms.RandomCrop(224), transforms.GaussianBlur(kernel_size=3), transforms.ToTensor(), transforms.Normalize(*norm)])