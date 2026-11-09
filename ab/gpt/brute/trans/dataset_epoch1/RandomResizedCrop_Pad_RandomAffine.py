import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.86), ratio=(1.21, 2.57)),
    transforms.Pad(padding=5, fill=(254, 48, 219), padding_mode='edge'),
    transforms.RandomAffine(degrees=24, translate=(0.0, 0.18), scale=(0.89, 1.94), shear=(3.69, 9.78)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
