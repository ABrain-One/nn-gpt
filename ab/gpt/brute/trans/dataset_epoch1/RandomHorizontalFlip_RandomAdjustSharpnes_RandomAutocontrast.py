import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.42),
    transforms.RandomAdjustSharpness(sharpness_factor=1.01, p=0.15),
    transforms.RandomAutocontrast(p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
