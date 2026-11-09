import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=0.76, p=0.54),
    transforms.RandomAffine(degrees=7, translate=(0.05, 0.17), scale=(0.89, 1.3), shear=(2.82, 6.25)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
