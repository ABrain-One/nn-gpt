import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=23, translate=(0.09, 0.16), scale=(0.96, 1.77), shear=(4.13, 7.79)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.09, p=0.31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
