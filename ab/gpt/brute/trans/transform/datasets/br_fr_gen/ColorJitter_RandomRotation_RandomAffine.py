import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.87, contrast=1.19, saturation=0.94, hue=0.0),
    transforms.RandomRotation(degrees=0),
    transforms.RandomAffine(degrees=15, translate=(0.01, 0.19), scale=(0.81, 1.9), shear=(3.57, 5.99)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
