import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.19, 0.15), scale=(0.86, 1.41), shear=(1.61, 5.57)),
    transforms.ColorJitter(brightness=0.89, contrast=1.14, saturation=0.8, hue=0.09),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
