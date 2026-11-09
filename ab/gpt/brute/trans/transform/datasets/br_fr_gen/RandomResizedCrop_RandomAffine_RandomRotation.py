import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.92), ratio=(1.2, 2.81)),
    transforms.RandomAffine(degrees=0, translate=(0.19, 0.13), scale=(1.12, 1.44), shear=(3.61, 8.97)),
    transforms.RandomRotation(degrees=2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
