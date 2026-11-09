import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.7), transforms.RandomPosterize(bits=7, p=0.14), transforms.RandomAffine(degrees=1, translate=(0.19, 0.1), scale=(1.13, 1.64), shear=(1.24, 8.41)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])