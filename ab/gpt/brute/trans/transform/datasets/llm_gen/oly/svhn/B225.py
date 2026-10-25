import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.8, 1.3)), transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.5, 1.5))], p=0.5), transforms.RandomAdjustSharpness(1.5, p=0.35), transforms.RandomRotation(degrees=(-15, 15)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.429, 0.403, 0.454], std=[0.186, 0.187, 0.178])])