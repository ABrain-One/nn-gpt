import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.RandomAffine(degrees=15, translate=(0.01, 0.0), scale=(1.04, 1.24), shear=(3.69, 5.45)),
    transforms.ColorJitter(brightness=0.99, contrast=1.11, saturation=0.99, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
