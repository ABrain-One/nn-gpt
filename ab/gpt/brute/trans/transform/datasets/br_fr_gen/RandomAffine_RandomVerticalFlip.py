import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=16, translate=(0.01, 0.19), scale=(0.87, 1.77), shear=(2.92, 9.05)),
    transforms.RandomVerticalFlip(p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
