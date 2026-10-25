import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.Resize(40), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 1.0)), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])