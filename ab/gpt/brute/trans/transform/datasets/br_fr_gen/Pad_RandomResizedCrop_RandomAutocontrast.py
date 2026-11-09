import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(212, 9, 177), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.96), ratio=(0.8, 2.45)),
    transforms.RandomAutocontrast(p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
