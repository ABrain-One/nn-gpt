import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.8), ratio=(1.11, 1.59)),
    transforms.RandomInvert(p=0.18),
    transforms.RandomAffine(degrees=23, translate=(0.05, 0.13), scale=(0.99, 1.23), shear=(2.99, 5.44)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
