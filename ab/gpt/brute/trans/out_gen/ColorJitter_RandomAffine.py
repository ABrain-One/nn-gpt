import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.08, contrast=1.15, saturation=1.05, hue=0.09),
    transforms.RandomAffine(degrees=6, translate=(0.16, 0.11), scale=(0.91, 1.25), shear=(2.04, 5.12)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
