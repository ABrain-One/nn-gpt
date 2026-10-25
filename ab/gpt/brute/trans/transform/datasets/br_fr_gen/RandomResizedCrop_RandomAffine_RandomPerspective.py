import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.85), ratio=(0.83, 2.71)),
    transforms.RandomAffine(degrees=11, translate=(0.18, 0.02), scale=(0.86, 1.21), shear=(2.63, 7.45)),
    transforms.RandomPerspective(distortion_scale=0.14, p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
