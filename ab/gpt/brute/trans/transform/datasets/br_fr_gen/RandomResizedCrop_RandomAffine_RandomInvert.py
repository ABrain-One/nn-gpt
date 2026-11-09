import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.95), ratio=(1.05, 2.85)),
    transforms.RandomAffine(degrees=2, translate=(0.18, 0.12), scale=(0.8, 1.34), shear=(1.32, 9.44)),
    transforms.RandomInvert(p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
