import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomRotation(degrees=23),
    transforms.RandomAffine(degrees=2, translate=(0.2, 0.15), scale=(1.2, 1.53), shear=(2.9, 8.14)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
