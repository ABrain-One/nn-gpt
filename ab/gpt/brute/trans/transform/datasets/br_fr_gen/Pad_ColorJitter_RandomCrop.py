import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(193, 161, 153), padding_mode='constant'),
    transforms.ColorJitter(brightness=1.05, contrast=0.97, saturation=0.8, hue=0.05),
    transforms.RandomCrop(size=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
