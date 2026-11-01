import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomAffine(degrees=19, translate=(0.13, 0.1), scale=(0.86, 1.55), shear=(2.05, 6.44)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
