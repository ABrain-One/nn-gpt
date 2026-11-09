import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.92), ratio=(1.3, 1.77)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.54, p=0.28),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.18), scale=(0.97, 1.59), shear=(2.53, 8.49)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
