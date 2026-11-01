import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.96), ratio=(1.07, 2.92)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.29, p=0.84),
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])