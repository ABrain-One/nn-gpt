import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.9), ratio=(1.29, 2.61)),
    transforms.RandomAffine(degrees=13, translate=(0.01, 0.11), scale=(1.01, 1.92), shear=(0.82, 5.56)),
    transforms.RandomEqualize(p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
