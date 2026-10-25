import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.82), ratio=(1.29, 1.91)),
    transforms.RandomAffine(degrees=5, translate=(0.09, 0.11), scale=(0.96, 1.45), shear=(4.11, 8.71)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
