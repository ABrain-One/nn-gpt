import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.91), ratio=(0.93, 2.58)),
    transforms.RandomVerticalFlip(p=0.87),
    transforms.RandomAutocontrast(p=0.11),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
