import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=0.55, p=0.35),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.98), ratio=(1.15, 2.0)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
