import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.19, contrast=0.89, saturation=1.11, hue=0.05),
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.99), ratio=(1.0, 1.57)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
