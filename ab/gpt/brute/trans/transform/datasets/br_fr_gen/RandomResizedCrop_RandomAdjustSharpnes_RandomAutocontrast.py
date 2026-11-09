import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.85), ratio=(0.94, 1.82)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.96, p=0.37),
    transforms.RandomAutocontrast(p=0.53),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
