import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(38, 66, 30), padding_mode='constant'),
    transforms.RandomVerticalFlip(p=0.86),
    transforms.RandomAutocontrast(p=0.71),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
