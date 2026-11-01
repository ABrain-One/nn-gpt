import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(195, 236, 160), padding_mode='constant'),
    transforms.ColorJitter(brightness=1.02, contrast=0.94, saturation=0.87, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
