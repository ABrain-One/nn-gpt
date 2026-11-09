import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomRotation(degrees=7),
    transforms.RandomAffine(degrees=22, translate=(0.14, 0.02), scale=(1.13, 1.24), shear=(0.74, 9.22)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
