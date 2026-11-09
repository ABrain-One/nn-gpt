import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=1, translate=(0.1, 0.08), scale=(0.82, 1.8), shear=(4.93, 6.58)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.71, p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
