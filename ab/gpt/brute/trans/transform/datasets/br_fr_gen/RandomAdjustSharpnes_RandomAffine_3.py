import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.38, p=0.84),
    transforms.RandomAffine(degrees=6, translate=(0.13, 0.17), scale=(0.81, 1.52), shear=(4.36, 5.97)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
