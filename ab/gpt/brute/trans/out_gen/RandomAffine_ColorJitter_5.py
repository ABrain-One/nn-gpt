import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.09, 0.14), scale=(0.95, 1.71), shear=(4.82, 9.2)),
    transforms.ColorJitter(brightness=1.04, contrast=0.8, saturation=1.02, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
