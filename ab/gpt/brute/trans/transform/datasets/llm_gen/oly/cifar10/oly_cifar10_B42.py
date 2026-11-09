import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.RandomPerspective(distortion_scale=0.3, p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Pad(padding=2, fill=(108, 134, 41), padding_mode='symmetric'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])