import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.6),
    transforms.RandomAffine(degrees=19, translate=(0.14, 0.03), scale=(1.13, 1.21), shear=(1.31, 7.12)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.07, p=0.85),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
