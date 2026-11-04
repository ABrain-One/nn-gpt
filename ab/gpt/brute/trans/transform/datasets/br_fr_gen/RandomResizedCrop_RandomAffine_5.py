import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.93), ratio=(1.18, 2.25)),
    transforms.RandomAffine(degrees=23, translate=(0.14, 0.08), scale=(0.97, 1.57), shear=(0.93, 6.86)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
