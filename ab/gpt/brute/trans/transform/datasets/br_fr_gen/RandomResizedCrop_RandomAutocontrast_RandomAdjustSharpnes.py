import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.92), ratio=(1.02, 1.38)),
    transforms.RandomAutocontrast(p=0.42),
    transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
