import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.08, contrast=1.04, saturation=0.87, hue=0.0),
    transforms.Pad(padding=2, fill=(63, 38, 167), padding_mode='edge'),
    transforms.RandomVerticalFlip(p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
