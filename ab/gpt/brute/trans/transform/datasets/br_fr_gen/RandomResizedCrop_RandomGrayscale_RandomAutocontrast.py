import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.87), ratio=(1.31, 1.77)),
    transforms.RandomGrayscale(p=0.13),
    transforms.RandomAutocontrast(p=0.6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
