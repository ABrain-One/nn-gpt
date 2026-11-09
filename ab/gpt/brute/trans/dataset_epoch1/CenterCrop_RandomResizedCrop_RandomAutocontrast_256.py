import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.96), ratio=(0.81, 2.23)),
    transforms.RandomAutocontrast(p=0.87),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
