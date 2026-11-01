import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.24), transforms.RandomAutocontrast(p=0.8), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])