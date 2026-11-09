import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.91), ratio=(1.02, 1.32)),
    transforms.RandomAutocontrast(p=0.62),
    transforms.RandomEqualize(p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])