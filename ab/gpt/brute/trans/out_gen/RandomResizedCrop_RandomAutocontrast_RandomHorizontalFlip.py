import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.94), ratio=(1.15, 1.46)),
    transforms.RandomAutocontrast(p=0.73),
    transforms.RandomHorizontalFlip(p=0.59),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
