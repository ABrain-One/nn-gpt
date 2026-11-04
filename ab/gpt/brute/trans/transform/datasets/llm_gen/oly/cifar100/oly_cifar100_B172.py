import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.1), shear=(-5, 5)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1))], p=0.3), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4971, 0.4865, 0.441], std=[0.247, 0.2435, 0.2616])])