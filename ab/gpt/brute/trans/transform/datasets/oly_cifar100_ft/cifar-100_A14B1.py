import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomPerspective(distortion_scale=0.15, p=0.36), transforms.RandomResizedCrop(size=32, scale=(0.7, 0.88), ratio=(1.17, 1.72)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])