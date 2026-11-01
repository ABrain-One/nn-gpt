import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(24), transforms.RandomGrayscale(p=0.68), transforms.RandomResizedCrop(32, scale=(0.53, 0.97), ratio=(1.33, 1.88)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])