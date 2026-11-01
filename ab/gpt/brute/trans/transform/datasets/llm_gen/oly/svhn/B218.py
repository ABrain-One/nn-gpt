import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(32), transforms.RandomCrop(31), transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(p=0.6), transforms.ColorJitter(brightness=0.93, contrast=1.02, saturation=0.82, hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])