import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.ColorJitter(brightness=1.08, contrast=1.08, saturation=0.85, hue=0.01),
    transforms.RandomAffine(degrees=24, translate=(0.06, 0.13), scale=(0.84, 1.41), shear=(1.92, 7.85)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
