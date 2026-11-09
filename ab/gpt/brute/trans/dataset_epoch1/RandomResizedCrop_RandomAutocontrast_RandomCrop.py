import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.88), ratio=(1.25, 2.41)),
    transforms.RandomAutocontrast(p=0.27),
    transforms.RandomCrop(size=32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
