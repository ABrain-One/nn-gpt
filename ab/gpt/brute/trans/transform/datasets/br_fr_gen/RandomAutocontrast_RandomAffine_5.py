import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.68),
    transforms.RandomAffine(degrees=26, translate=(0.09, 0.12), scale=(0.85, 1.29), shear=(2.42, 5.07)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
