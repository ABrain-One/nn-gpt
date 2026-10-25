import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomInvert(p=0.2), transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-5, 5), resample=transforms.InterpolationMode.BICUBIC, fill=(122, 117, 104)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])