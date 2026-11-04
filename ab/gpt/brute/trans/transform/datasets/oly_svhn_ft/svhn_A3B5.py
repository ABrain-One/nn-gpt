import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(0.9, 1.67)), transforms.RandomGrayscale(p=0.39), transforms.RandomPerspective(distortion_scale=0.12, p=0.39), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])