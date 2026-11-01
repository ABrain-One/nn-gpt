import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(196, 163, 161), padding_mode='edge'),
    transforms.RandomCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.92), ratio=(0.87, 1.4)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])