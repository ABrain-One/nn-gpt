import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(170, 0, 72), padding_mode='edge'),
    transforms.RandomVerticalFlip(p=0.21),
    transforms.ColorJitter(brightness=0.84, contrast=0.99, saturation=0.96, hue=0.04),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
