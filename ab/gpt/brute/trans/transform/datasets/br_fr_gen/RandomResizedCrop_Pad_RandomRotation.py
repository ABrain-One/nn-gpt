import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.95), ratio=(0.96, 2.8)),
    transforms.Pad(padding=3, fill=(221, 195, 64), padding_mode='reflect'),
    transforms.RandomRotation(degrees=12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
