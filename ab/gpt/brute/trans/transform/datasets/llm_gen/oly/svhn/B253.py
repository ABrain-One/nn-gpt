import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 2.0), ratio=(0.5, 2.0)), transforms.RandomAutocontrast(p=0.12), transforms.RandomPerspective(distortion_scale=0.19, p=0.63), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])