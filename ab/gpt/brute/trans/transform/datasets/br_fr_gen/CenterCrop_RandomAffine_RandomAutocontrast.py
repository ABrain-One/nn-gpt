import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomAffine(degrees=30, translate=(0.15, 0.18), scale=(1.05, 1.95), shear=(0.36, 5.11)),
    transforms.RandomAutocontrast(p=0.46),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
