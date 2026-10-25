import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.85), transforms.RandomAutocontrast(p=0.87), transforms.RandomEqualize(p=0.78), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])