import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.67),
    transforms.RandomAffine(degrees=0, translate=(0.11, 0.04), scale=(0.84, 1.95), shear=(0.84, 9.18)),
    transforms.RandomRotation(degrees=6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
