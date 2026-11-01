import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.24), transforms.RandomResizedCrop(32, scale=(0.72, 0.92), ratio=(1.32, 1.68)), transforms.RandomPerspective(distortion_scale=0.25, p=0.43), transforms.RandomGrayscale(p=0.23), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])