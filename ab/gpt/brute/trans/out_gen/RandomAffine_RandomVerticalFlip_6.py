import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.13, 0.02), scale=(0.98, 1.78), shear=(2.8, 6.05)),
    transforms.RandomVerticalFlip(p=0.77),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
