import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.67),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.81), ratio=(0.83, 2.86)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.89, p=0.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
