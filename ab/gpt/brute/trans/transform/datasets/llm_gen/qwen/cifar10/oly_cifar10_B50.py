import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(degrees=15), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.75, 1.25)), transforms.RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), transforms.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 3.33), value=(5, 216, 154)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])