import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomAffine(degrees=0, translate=(0.13, 0.17), scale=(1.0, 1.84), shear=(0.09, 8.94)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.92, p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
