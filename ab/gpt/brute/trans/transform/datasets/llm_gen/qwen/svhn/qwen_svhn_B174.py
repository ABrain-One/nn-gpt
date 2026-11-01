import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.3), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01), transforms.RandomGrayscale(p=0.1), transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(*norm)])