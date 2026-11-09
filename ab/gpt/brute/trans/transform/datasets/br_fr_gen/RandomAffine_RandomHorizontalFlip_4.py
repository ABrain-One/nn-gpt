import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.18, 0.13), scale=(0.87, 1.27), shear=(1.41, 8.11)),
    transforms.RandomHorizontalFlip(p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
