import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.13, 0.11), scale=(0.93, 1.21), shear=(1.43, 9.51)),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.85), ratio=(1.04, 1.95)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
