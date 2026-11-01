import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.73),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.85), ratio=(1.29, 2.47)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
