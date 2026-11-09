import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(153, 107, 255), padding_mode='constant'),
    transforms.ColorJitter(brightness=0.8, contrast=0.92, saturation=0.81, hue=0.04),
    transforms.RandomRotation(degrees=8),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
