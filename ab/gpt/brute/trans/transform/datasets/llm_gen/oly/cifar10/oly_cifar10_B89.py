import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=(-10, 10)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*norm)])