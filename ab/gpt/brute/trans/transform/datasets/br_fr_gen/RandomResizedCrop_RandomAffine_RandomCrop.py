import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.99), ratio=(0.98, 2.71)),
    transforms.RandomAffine(degrees=19, translate=(0.08, 0.11), scale=(1.1, 1.29), shear=(3.55, 5.14)),
    transforms.RandomCrop(size=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
