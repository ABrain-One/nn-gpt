import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=1.07, p=0.28), transforms.RandomAffine(degrees=19, translate=(0.09, 0.12), scale=(1.1, 1.19), shear=(3.21, 5.41)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])