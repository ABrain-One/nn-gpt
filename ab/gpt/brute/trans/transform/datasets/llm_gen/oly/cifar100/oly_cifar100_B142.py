import torch
import torchvision.transforms as transforms
from PIL import Image

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.8, 1.2)), transforms.GaussianBlur(5, sigma=(0.5, 1.5)), transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5.0, 5.0)), transforms.RandomVerticalFlip(p=0.2), transforms.ToTensor(), transforms.Normalize(**norm)])