import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(1.31, 2.06)), transforms.RandomPosterize(bits=8, p=0.54), transforms.RandomAffine(degrees=13, translate=(0.03, 0.08), scale=(0.98, 1.89), shear=(1.92, 9.05)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])