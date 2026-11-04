import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.8, 2.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.4, 0.4, 0.4, 0.2), transforms.RandomPerspective(distortion_scale=0.2, p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])