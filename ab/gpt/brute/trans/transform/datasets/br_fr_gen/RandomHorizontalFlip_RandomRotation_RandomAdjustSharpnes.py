import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.26),
    transforms.RandomRotation(degrees=24),
    transforms.RandomAdjustSharpness(sharpness_factor=0.61, p=0.71),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
