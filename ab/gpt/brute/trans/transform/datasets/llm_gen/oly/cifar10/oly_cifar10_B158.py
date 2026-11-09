import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomGrayscale(p=0.2), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])