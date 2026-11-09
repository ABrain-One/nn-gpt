import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 0.85), ratio=(0.75, 2.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.9, contrast=0.7, saturation=0.7, hue=0.0), transforms.Grayscale(num_output_channels=3), transforms.CenterCrop(size=30), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])