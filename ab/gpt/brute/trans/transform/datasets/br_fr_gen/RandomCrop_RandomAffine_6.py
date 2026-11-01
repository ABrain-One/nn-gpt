import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomAffine(degrees=5, translate=(0.09, 0.05), scale=(1.07, 1.3), shear=(4.23, 5.87)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
