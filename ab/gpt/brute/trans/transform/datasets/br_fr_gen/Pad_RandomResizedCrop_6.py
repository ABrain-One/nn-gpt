import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(249, 118, 63), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(1.0, 1.95)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
