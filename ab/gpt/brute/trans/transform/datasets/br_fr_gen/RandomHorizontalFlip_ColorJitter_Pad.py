import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.24),
    transforms.ColorJitter(brightness=1.14, contrast=0.89, saturation=1.11, hue=0.01),
    transforms.Pad(padding=5, fill=(68, 154, 69), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
