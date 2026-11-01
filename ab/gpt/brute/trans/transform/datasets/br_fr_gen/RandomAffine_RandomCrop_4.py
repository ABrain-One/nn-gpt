import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=16, translate=(0.03, 0.02), scale=(0.82, 1.7), shear=(4.46, 8.59)),
    transforms.RandomCrop(size=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
