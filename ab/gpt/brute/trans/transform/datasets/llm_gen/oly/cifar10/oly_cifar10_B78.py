import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.6), transforms.RandomVerticalFlip(p=0.2), transforms.RandomRotation(degrees=(-10, +10)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=0.5), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])