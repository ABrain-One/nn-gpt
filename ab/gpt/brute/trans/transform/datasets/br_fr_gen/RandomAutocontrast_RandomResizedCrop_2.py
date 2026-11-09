import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.27),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.82), ratio=(1.12, 2.87)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
