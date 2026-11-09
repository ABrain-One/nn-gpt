import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomAffine(degrees=3, translate=(0.1, 0.15), scale=(0.88, 1.2), shear=(3.64, 9.55)),
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.83), ratio=(0.96, 1.61)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
