import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.12),
    transforms.RandomPosterize(bits=6, p=0.36),
    transforms.RandomAffine(degrees=11, translate=(0.14, 0.13), scale=(0.95, 1.84), shear=(1.91, 9.05)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
