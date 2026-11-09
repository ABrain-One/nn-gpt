import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.89), ratio=(0.93, 2.46)),
    transforms.RandomAutocontrast(p=0.3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
