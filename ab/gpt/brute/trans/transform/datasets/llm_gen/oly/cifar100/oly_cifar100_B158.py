import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.64), transforms.RandomRotation(degrees=7), transforms.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.7, hue=0.05), transforms.RandomPosterize(bits=5, p=0.69), transforms.RandomSolarize(threshold=30, p=0.3), transforms.Pad(padding=2, fill=(111, 235, 143), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])