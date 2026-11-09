import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(186, 21, 27), padding_mode='edge'),
    transforms.RandomRotation(degrees=22),
    transforms.ColorJitter(brightness=0.84, contrast=0.8, saturation=0.81, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
