import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomAffine(degrees=14, translate=(0.08, 0.15), scale=(0.9, 1.38), shear=(0.31, 7.02)),
    transforms.Pad(padding=3, fill=(50, 21, 201), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
