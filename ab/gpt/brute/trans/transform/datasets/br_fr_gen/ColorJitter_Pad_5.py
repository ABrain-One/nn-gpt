import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.12, contrast=1.1, saturation=1.04, hue=0.06),
    transforms.Pad(padding=5, fill=(96, 74, 163), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
