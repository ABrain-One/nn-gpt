import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=30, translate=(0.06, 0.15), scale=(1.04, 1.24), shear=(0.12, 9.15)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
