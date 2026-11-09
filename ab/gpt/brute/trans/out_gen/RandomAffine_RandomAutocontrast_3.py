import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=14, translate=(0.18, 0.03), scale=(0.98, 1.4), shear=(3.35, 8.82)),
    transforms.RandomAutocontrast(p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
