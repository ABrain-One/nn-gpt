import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.85), ratio=(1.14, 1.79)),
    transforms.RandomRotation(degrees=23),
    transforms.Pad(padding=4, fill=(2, 226, 181), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
