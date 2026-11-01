import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=10), transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8), transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=3, sigma=1.0, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])