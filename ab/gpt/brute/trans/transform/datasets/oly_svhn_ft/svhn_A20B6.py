import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.76, 0.87), ratio=(1.04, 1.95)), transforms.RandomPerspective(distortion_scale=0.18, p=0.57), transforms.RandomCrop(size=29), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])