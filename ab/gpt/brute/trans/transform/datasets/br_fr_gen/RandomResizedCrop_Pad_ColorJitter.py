import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.5, 0.84), ratio=(0.79, 2.45)),
    transforms.Pad(padding=0, fill=(189, 122, 85), padding_mode='edge'),
    transforms.ColorJitter(brightness=1.09, contrast=0.91, saturation=1.17, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
