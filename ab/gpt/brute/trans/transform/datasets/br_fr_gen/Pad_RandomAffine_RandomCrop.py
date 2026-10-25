import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(186, 58, 81), padding_mode='constant'),
    transforms.RandomAffine(degrees=29, translate=(0.11, 0.13), scale=(1.1, 1.24), shear=(3.97, 9.38)),
    transforms.RandomCrop(size=32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
