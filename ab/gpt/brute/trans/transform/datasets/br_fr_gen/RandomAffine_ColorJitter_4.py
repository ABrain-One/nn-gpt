import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=21, translate=(0.03, 0.18), scale=(0.8, 1.85), shear=(2.24, 6.17)),
    transforms.ColorJitter(brightness=1.14, contrast=1.07, saturation=1.19, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
