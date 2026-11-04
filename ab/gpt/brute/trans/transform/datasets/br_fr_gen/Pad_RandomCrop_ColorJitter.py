import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(196, 132, 88), padding_mode='reflect'),
    transforms.RandomCrop(size=30),
    transforms.ColorJitter(brightness=1.14, contrast=1.03, saturation=1.14, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
