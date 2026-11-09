import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.11, p=0.6),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.98), ratio=(1.15, 2.79)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
