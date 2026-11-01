import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((64, 64)), transforms.RandomAutocontrast(p=0.9), transforms.RandomInvert(p=0.9), transforms.RandomSolarize(threshold=128, p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])