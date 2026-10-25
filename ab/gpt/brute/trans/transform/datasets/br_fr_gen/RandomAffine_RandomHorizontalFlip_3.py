import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=28, translate=(0.18, 0.18), scale=(1.19, 1.22), shear=(0.66, 7.35)),
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
