import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.87),
    transforms.Pad(padding=5, fill=(165, 166, 84), padding_mode='edge'),
    transforms.RandomAutocontrast(p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
