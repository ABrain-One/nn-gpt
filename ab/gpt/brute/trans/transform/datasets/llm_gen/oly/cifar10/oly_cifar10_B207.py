import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=(-5, 5)), transforms.Pad(4, fill=(125.3, 118.8, 107.3), padding_mode='constant'), transforms.RandomCrop(32), transforms.Resize(224), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])