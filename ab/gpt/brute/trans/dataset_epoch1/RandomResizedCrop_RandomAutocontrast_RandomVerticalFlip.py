import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.97), ratio=(0.95, 1.7)),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomVerticalFlip(p=0.49),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
