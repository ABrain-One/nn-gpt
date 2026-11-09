import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.03, contrast=0.94, saturation=1.0, hue=0.05),
    transforms.RandomRotation(degrees=23),
    transforms.Pad(padding=5, fill=(5, 130, 170), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
