import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=23, translate=(0.13, 0.18), scale=(1.15, 1.74), shear=(2.34, 7.41)),
    transforms.RandomVerticalFlip(p=0.24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
