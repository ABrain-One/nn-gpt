import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=0.95, p=0.25), transforms.RandomAffine(degrees=0, translate=(0.01, 0.15), scale=(0.93, 1.3), shear=(0.46, 6.42)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])