import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.16, contrast=0.99, saturation=1.1, hue=0.09),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.89), ratio=(0.94, 2.82)),
    transforms.RandomHorizontalFlip(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
