import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=32), transforms.RandomAdjustSharpness(sharpness_factor=0.54, p=0.11), transforms.RandomAffine(degrees=13, translate=(0.04, 0.11), scale=(1.11, 1.23), shear=(4.19, 6.05)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])