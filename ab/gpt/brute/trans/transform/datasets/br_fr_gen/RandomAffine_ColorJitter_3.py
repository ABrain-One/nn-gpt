import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=13, translate=(0.18, 0.17), scale=(1.16, 1.38), shear=(0.62, 7.8)),
    transforms.ColorJitter(brightness=0.88, contrast=0.85, saturation=0.92, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
