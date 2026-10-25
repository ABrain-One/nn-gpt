import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(43, 79, 30), padding_mode='symmetric'), transforms.RandomRotation(degrees=(-10, 10)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomPerspective(distortion_scale=0.15, p=0.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])