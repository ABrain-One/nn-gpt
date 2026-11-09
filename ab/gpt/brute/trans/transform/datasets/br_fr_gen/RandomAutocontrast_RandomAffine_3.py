import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.25),
    transforms.RandomAffine(degrees=23, translate=(0.09, 0.0), scale=(0.95, 1.53), shear=(4.93, 8.2)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
