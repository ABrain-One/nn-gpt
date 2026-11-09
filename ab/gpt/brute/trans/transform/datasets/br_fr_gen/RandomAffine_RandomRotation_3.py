import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.05), scale=(0.94, 1.22), shear=(4.56, 6.56)),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
