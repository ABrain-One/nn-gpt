import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.98), ratio=(0.92, 1.83)),
    transforms.RandomInvert(p=0.38),
    transforms.RandomAdjustSharpness(sharpness_factor=0.89, p=0.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
