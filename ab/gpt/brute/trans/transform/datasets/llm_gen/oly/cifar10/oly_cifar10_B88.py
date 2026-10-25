import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomInvert(p=0.83), transforms.RandomAutocontrast(p=0.77), transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(*norm)])