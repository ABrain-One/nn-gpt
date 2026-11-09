import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.5, 0.99), ratio=(1.02, 2.41)), transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.02), transforms.RandomHorizontalFlip(p=0.5), transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])