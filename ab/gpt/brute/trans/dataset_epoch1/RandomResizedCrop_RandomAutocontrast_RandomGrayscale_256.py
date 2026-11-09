import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.96), ratio=(0.85, 2.72)),
    transforms.RandomAutocontrast(p=0.66),
    transforms.RandomGrayscale(p=0.54),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
