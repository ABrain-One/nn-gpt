import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.8, contrast=1.13, saturation=0.88, hue=0.09),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.98), ratio=(1.3, 2.14)),
    transforms.RandomAffine(degrees=21, translate=(0.14, 0.06), scale=(0.9, 1.55), shear=(1.19, 6.74)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
