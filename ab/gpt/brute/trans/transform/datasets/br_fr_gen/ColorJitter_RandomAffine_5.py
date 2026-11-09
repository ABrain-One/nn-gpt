import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.85, contrast=1.01, saturation=1.06, hue=0.03),
    transforms.RandomAffine(degrees=28, translate=(0.13, 0.11), scale=(0.83, 1.28), shear=(4.45, 5.2)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
