import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.85, contrast=0.83, saturation=0.8, hue=0.09),
    transforms.RandomAffine(degrees=11, translate=(0.15, 0.17), scale=(0.86, 1.23), shear=(1.71, 8.36)),
    transforms.RandomHorizontalFlip(p=0.75),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
