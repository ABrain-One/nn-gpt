import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.14, p=0.57),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.85), ratio=(1.29, 1.37)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
