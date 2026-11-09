import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(157, 242, 108), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.35, p=0.72),
    transforms.ColorJitter(brightness=1.2, contrast=1.12, saturation=1.11, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
