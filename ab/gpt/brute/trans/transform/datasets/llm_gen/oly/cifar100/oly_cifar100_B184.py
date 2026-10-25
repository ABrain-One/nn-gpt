import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.75, 1.333)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])