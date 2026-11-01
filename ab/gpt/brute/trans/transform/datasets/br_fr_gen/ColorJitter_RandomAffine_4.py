import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.06, contrast=1.02, saturation=0.99, hue=0.01),
    transforms.RandomAffine(degrees=2, translate=(0.03, 0.04), scale=(0.86, 1.28), shear=(3.15, 7.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
