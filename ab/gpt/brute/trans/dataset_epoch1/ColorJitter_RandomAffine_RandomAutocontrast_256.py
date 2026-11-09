import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.99, contrast=1.1, saturation=1.09, hue=0.03),
    transforms.RandomAffine(degrees=12, translate=(0.16, 0.15), scale=(1.11, 1.8), shear=(0.84, 9.91)),
    transforms.RandomAutocontrast(p=0.69),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
