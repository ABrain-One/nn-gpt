import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.98), ratio=(0.77, 2.38)),
    transforms.RandomEqualize(p=0.6),
    transforms.RandomAutocontrast(p=0.74),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
