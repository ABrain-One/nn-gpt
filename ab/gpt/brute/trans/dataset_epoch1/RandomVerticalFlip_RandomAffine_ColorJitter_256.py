import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomAffine(degrees=23, translate=(0.08, 0.05), scale=(0.85, 1.47), shear=(0.84, 9.18)),
    transforms.ColorJitter(brightness=0.9, contrast=0.92, saturation=0.96, hue=0.05),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
