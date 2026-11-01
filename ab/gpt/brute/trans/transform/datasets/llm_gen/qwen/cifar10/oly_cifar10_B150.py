import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)), transforms.RandomRotation(degrees=(-15, 15)), transforms.ToTensor(), transforms.Normalize(*norm)])