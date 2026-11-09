import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.28),
    transforms.ColorJitter(brightness=1.01, contrast=1.08, saturation=0.94, hue=0.08),
    transforms.Pad(padding=1, fill=(164, 187, 7), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
