import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.53),
    transforms.RandomRotation(degrees=8),
    transforms.RandomAffine(degrees=6, translate=(0.09, 0.13), scale=(1.2, 1.23), shear=(4.13, 5.35)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
