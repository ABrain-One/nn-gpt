import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=16, translate=(0.17, 0.02), scale=(0.83, 1.89), shear=(0.46, 7.83)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
