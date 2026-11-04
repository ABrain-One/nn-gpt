import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.33), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.45), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.5, 2.5), p=0.17), transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5), p=0.25), transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])