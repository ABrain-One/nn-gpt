import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.RandomGrayscale(p=0.5), transforms.RandomHorizontalFlip(p=0.7), transforms.RandomVerticalFlip(p=0.3), transforms.RandomRotation(degrees=(-10, +10)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5), p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])