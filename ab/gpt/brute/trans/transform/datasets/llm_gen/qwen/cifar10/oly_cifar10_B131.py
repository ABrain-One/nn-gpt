import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomGrayscale(p=0.2), transforms.RandomPerspective(distortion_scale=0.2, p=0.2), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2), ratio=(0.8, 1.2)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])