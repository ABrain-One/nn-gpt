import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=1.15, saturation=1.01, hue=0.01),
    transforms.RandomGrayscale(p=0.18),
    transforms.Pad(padding=2, fill=(105, 202, 151), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])