import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.0, 0.06), scale=(1.0, 1.59), shear=(0.69, 9.51)),
    transforms.RandomRotation(degrees=21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
