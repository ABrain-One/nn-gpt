import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.84), ratio=(1.18, 2.57)),
    transforms.RandomAffine(degrees=13, translate=(0.02, 0.08), scale=(1.14, 1.76), shear=(1.97, 8.02)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
