import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomEqualize(p=0.22),
    transforms.RandomAffine(degrees=16, translate=(0.1, 0.19), scale=(0.84, 1.7), shear=(4.7, 6.75)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
