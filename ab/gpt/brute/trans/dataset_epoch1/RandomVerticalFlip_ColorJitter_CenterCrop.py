import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.12),
    transforms.ColorJitter(brightness=1.16, contrast=0.92, saturation=1.05, hue=0.04),
    transforms.CenterCrop(size=31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
