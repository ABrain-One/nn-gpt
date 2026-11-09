import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.41), transforms.RandomRotation(degrees=12), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])