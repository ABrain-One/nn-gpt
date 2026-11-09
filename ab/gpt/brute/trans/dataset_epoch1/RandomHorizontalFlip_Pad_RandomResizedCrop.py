import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.Pad(padding=4, fill=(100, 158, 67), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.81), ratio=(0.89, 1.43)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
