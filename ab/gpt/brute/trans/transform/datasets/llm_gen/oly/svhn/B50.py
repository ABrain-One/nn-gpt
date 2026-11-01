import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=1.09, contrast=1.12, saturation=1.11, hue=0.1), transforms.RandomRotation(degrees=26), transforms.CenterCrop(31), transforms.RandomPerspective(distortion_scale=0.13, p=0.67), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])