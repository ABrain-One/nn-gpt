import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.14), transforms.ColorJitter(brightness=0.89, contrast=1.04, saturation=0.94, hue=0.03), transforms.RandomRotation(degrees=25), transforms.RandomAffine(degrees=29, translate=(0.02, 0.02), scale=(1.14, 1.59), shear=(3.47, 5.56)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])