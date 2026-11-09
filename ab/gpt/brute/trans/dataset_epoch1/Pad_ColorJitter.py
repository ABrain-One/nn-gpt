import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(158, 27, 136), padding_mode='constant'),
    transforms.ColorJitter(brightness=0.82, contrast=0.9, saturation=0.95, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
