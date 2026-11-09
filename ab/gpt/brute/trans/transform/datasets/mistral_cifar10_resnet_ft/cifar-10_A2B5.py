import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.82), transforms.RandomAffine(degrees=14, translate=(0.1, 0.04), scale=(1.07, 1.52), shear=(4.03, 9.45)), transforms.RandomAutocontrast(p=0.78), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])