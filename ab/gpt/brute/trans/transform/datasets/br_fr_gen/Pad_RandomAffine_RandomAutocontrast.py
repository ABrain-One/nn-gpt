import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(251, 11, 39), padding_mode='reflect'),
    transforms.RandomAffine(degrees=12, translate=(0.03, 0.08), scale=(1.01, 1.57), shear=(0.57, 5.34)),
    transforms.RandomAutocontrast(p=0.5),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
