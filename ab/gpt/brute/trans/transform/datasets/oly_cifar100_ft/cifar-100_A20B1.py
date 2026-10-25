import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=0.55, p=0.3), transforms.RandomAffine(degrees=13, translate=(0.09, 0.04), scale=(1.12, 1.6), shear=(4.14, 8.83)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])