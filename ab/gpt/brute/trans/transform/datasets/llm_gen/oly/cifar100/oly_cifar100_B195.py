import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.55), transforms.RandomVerticalFlip(p=0.83), transforms.RandomSolarize(threshold=140, p=0.78), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomRotation(degrees=(-10, 10)), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(**norm)])