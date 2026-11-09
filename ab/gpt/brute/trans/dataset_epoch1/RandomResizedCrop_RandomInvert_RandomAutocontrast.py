import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.88), ratio=(0.89, 2.39)),
    transforms.RandomInvert(p=0.54),
    transforms.RandomAutocontrast(p=0.25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
