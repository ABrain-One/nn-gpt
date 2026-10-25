import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(6), transforms.ColorJitter(brightness=0.8, contrast=1.1, saturation=1.1, hue=0.05), transforms.RandomVerticalFlip(p=0.2), transforms.RandomPerspective(distortion_scale=0.27, p=0.22), transforms.Resize(64), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(*norm)])