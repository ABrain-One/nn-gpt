import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=25, translate=(0.18, 0.12), scale=(0.96, 1.88), shear=(0.6, 8.64)),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.91), ratio=(1.22, 2.61)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
