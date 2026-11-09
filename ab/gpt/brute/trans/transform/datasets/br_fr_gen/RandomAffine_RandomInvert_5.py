import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=14, translate=(0.0, 0.08), scale=(1.14, 1.98), shear=(1.19, 7.66)),
    transforms.RandomInvert(p=0.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
