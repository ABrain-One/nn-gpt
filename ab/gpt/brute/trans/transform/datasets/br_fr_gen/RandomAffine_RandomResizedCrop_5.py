import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=3, translate=(0.07, 0.03), scale=(1.06, 1.66), shear=(2.58, 7.59)),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.85), ratio=(0.89, 1.78)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
