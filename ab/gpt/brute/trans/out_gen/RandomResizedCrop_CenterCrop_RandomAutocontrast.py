import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 1.0), ratio=(1.13, 1.91)),
    transforms.CenterCrop(size=29),
    transforms.RandomAutocontrast(p=0.74),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
