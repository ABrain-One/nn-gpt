import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.98, contrast=1.18, saturation=1.13, hue=0.07),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.89), ratio=(0.94, 2.9)),
    transforms.RandomRotation(degrees=4),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
