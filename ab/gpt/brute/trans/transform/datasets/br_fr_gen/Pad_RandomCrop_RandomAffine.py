import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(202, 39, 176), padding_mode='constant'),
    transforms.RandomCrop(size=32),
    transforms.RandomAffine(degrees=8, translate=(0.09, 0.04), scale=(1.1, 1.93), shear=(0.41, 6.61)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
