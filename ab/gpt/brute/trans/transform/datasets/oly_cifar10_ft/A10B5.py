import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(63, 38, 125), padding_mode='constant'),
    transforms.RandomVerticalFlip(p=0.83),
    transforms.RandomAutocontrast(p=0.76, reg=0.7),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])