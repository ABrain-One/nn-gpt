import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.57, 0.84), ratio=(1.01, 1.83)), transforms.RandomPerspective(distortion_scale=0.22, p=0.67), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])