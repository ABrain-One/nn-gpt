import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.3), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])