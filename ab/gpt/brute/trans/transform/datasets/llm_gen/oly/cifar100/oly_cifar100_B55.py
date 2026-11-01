import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(162, 51, 17), padding_mode='edge'), transforms.RandomRotation(degrees=(-5, 5)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(p=0.25), transforms.RandomVerticalFlip(p=0.1), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])