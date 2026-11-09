import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=(208, 208, 208, 208), mode='constant', fill=0), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=5, sigma=(0.4, 1.3)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])