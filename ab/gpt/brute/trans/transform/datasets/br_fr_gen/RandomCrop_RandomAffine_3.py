import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.RandomAffine(degrees=10, translate=(0.12, 0.15), scale=(1.01, 1.22), shear=(0.5, 6.84)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
