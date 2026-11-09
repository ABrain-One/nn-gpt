import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(*norm)])