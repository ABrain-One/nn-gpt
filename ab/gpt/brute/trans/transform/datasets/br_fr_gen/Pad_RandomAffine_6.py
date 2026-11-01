import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(28, 250, 81), padding_mode='reflect'),
    transforms.RandomAffine(degrees=2, translate=(0.18, 0.03), scale=(0.9, 1.39), shear=(4.21, 5.54)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
