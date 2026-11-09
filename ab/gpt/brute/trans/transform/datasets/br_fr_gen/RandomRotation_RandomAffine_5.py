import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=3),
    transforms.RandomAffine(degrees=16, translate=(0.05, 0.07), scale=(0.85, 1.46), shear=(1.79, 9.38)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
