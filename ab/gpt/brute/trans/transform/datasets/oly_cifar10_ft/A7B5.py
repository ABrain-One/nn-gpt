import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.96), ratio=(1.09, 1.46)),
    transforms.RandomAutocontrast(p=0.8),
    transforms.RandomGrayscale(p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])