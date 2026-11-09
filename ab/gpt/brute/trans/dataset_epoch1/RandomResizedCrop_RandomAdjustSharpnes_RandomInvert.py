import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(0.75, 1.57)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.89, p=0.87),
    transforms.RandomInvert(p=0.58),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
