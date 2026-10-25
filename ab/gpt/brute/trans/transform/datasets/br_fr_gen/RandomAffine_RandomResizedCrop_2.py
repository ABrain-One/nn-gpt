import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.16, 0.08), scale=(0.83, 1.58), shear=(1.1, 9.98)),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.82), ratio=(1.19, 2.88)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
