import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.18),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.83), ratio=(0.98, 2.2)),
    transforms.RandomAutocontrast(p=0.73),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
