import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.09, contrast=1.12, saturation=0.8, hue=0.09),
    transforms.RandomAffine(degrees=24, translate=(0.05, 0.11), scale=(1.18, 1.33), shear=(1.68, 8.56)),
    transforms.RandomRotation(degrees=14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
