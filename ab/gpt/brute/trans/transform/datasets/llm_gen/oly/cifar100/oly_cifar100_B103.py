import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(64), transforms.RandomRotation(15), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.2), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])