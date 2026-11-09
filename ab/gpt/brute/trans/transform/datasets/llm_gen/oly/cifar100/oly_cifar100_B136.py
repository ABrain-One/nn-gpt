import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.Solarize(threshold=180)], p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])