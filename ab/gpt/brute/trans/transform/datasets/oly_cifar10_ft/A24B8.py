import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=0.57, p=0.71),
    transforms.RandomAffine(degrees=10, translate=(0.09, 0.09), scale=(1.0, 1.95), shear=(1.92, 8.85)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])