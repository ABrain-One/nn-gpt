import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.3), transforms.RandomRotation(degrees=(-15, 15)), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])