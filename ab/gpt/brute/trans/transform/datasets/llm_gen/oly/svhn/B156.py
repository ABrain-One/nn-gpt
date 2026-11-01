import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomPosterize(bits=5, p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.4388, 0.469, 0.4745], std=[0.2531, 0.2355, 0.2451])])