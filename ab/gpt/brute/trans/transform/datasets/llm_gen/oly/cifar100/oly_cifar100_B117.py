import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(128), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.5, 1.5)), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(**norm)])