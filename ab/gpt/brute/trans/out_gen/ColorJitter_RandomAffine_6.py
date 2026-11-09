import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.97, contrast=0.89, saturation=0.83, hue=0.01),
    transforms.RandomAffine(degrees=30, translate=(0.02, 0.09), scale=(0.97, 1.68), shear=(0.56, 8.85)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
