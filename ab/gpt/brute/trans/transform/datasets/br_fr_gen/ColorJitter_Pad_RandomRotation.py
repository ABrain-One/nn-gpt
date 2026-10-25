import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.99, contrast=0.82, saturation=1.1, hue=0.07),
    transforms.Pad(padding=4, fill=(167, 163, 41), padding_mode='edge'),
    transforms.RandomRotation(degrees=19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
