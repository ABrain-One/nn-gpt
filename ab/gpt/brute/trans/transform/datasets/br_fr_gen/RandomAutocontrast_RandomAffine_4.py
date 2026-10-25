import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.8),
    transforms.RandomAffine(degrees=22, translate=(0.07, 0.19), scale=(1.03, 1.59), shear=(1.38, 7.67)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
