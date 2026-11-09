import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.84), ratio=(0.98, 1.45)),
    transforms.ColorJitter(brightness=0.91, contrast=1.0, saturation=1.13, hue=0.04),
    transforms.Pad(padding=0, fill=(69, 159, 63), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
