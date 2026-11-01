import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.9, contrast=1.03, saturation=0.98, hue=0.05),
    transforms.RandomAffine(degrees=16, translate=(0.16, 0.09), scale=(0.84, 1.61), shear=(3.24, 7.22)),
    transforms.RandomCrop(size=31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
