import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.4), transforms.RandomAffine(degrees=10, translate=(0.1, 0.05), scale=(0.8, 1.2), shear=(-5.0, 5.0)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomPosterize(bits=5, p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4913997, 0.48215845, 0.4465309], std=[0.24703225, 0.2434851, 0.26158083])])