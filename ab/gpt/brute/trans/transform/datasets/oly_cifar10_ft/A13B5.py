import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(118, 159, 46), padding_mode='edge'),
    transforms.ColorJitter(brightness=0.82, contrast=1.14, saturation=0.95, hue=0.04),
    transforms.RandomCrop(size=31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])