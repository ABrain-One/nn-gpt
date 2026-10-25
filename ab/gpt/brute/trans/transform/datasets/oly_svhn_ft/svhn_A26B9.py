import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPosterize(bits=7, p=0.38), transforms.RandomAffine(degrees=9, translate=(0.13, 0.11), scale=(1.03, 1.71), shear=(4.15, 5.52)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])