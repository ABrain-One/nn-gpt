<code>
import torch
import torchvision.transforms as transforms
from PIL import Image

def transform(norm):
    return transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
</code>