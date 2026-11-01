import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.98), ratio=(1.17, 2.13)),
    transforms.ColorJitter(brightness=0.87, contrast=1.14, saturation=1.17, hue=0.04),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.59, 1.53)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
