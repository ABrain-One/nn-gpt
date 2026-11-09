import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(68, 165, 51), padding_mode='constant'),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.91), ratio=(0.92, 2.92)),
    transforms.RandomRotation(degrees=9),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
