import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.81), ratio=(0.84, 1.99)),
    transforms.RandomHorizontalFlip(p=0.41),
    transforms.ColorJitter(brightness=1.15, contrast=0.8, saturation=0.83, hue=0.0),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
