import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=29, translate=(0.07, 0.02), scale=(0.89, 1.25), shear=(4.49, 9.24)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.81, p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
