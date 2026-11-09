import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.76),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.98), ratio=(1.05, 1.43)),
    transforms.ColorJitter(brightness=0.97, contrast=1.15, saturation=0.82, hue=0.05),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
