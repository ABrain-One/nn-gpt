import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(53, 22, 140), padding_mode='constant'),
    transforms.RandomHorizontalFlip(p=0.66),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.9), ratio=(0.8, 1.58)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
