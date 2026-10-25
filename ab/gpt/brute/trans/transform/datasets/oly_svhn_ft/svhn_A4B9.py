import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.4), transforms.RandomAutocontrast(p=0.39), transforms.RandomInvert(p=0.32), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])