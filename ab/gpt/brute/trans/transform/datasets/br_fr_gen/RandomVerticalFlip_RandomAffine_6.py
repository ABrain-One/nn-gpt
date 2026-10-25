import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.8),
    transforms.RandomAffine(degrees=10, translate=(0.13, 0.03), scale=(1.02, 1.72), shear=(1.9, 8.79)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
