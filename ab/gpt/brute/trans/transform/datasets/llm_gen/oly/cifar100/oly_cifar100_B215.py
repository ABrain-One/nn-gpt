import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.25), transforms.RandomCrop(28), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.75), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])