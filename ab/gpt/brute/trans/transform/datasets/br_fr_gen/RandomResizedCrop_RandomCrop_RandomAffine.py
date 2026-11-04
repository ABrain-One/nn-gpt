import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.82), ratio=(1.18, 1.44)),
    transforms.RandomCrop(size=25),
    transforms.RandomAffine(degrees=2, translate=(0.12, 0.13), scale=(1.03, 1.9), shear=(1.13, 9.66)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
