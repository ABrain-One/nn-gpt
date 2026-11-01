import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.86, contrast=1.16, saturation=0.96, hue=0.06),
    transforms.Pad(padding=2, fill=(135, 203, 132), padding_mode='constant'),
    transforms.RandomHorizontalFlip(p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
