import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.ToTensor(), transforms.Normalize(*norm)])