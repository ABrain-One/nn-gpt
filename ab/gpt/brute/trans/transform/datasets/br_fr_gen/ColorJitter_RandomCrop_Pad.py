import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.01, contrast=0.96, saturation=1.03, hue=0.03),
    transforms.RandomCrop(size=32),
    transforms.Pad(padding=2, fill=(23, 144, 84), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
