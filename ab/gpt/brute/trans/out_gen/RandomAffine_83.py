import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=29, translate=(0.18, 0.02), scale=(0.88, 1.28), shear=(0.82, 8.4)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
