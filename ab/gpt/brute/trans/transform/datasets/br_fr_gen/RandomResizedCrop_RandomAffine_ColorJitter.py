import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.9), ratio=(1.29, 1.51)),
    transforms.RandomAffine(degrees=21, translate=(0.17, 0.02), scale=(0.82, 1.92), shear=(3.33, 8.14)),
    transforms.ColorJitter(brightness=0.94, contrast=1.11, saturation=0.84, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
