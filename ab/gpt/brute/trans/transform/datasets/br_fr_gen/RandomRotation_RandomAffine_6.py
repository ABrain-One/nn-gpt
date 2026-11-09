import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=13),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.01), scale=(1.12, 1.53), shear=(3.48, 6.63)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
