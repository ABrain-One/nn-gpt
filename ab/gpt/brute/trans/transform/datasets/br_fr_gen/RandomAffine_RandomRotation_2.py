import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=6, translate=(0.2, 0.03), scale=(0.92, 1.3), shear=(3.55, 6.92)),
    transforms.RandomRotation(degrees=24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
