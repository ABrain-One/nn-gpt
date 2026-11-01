import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.85, contrast=1.0, saturation=0.9, hue=0.07),
    transforms.CenterCrop(size=25),
    transforms.Pad(padding=3, fill=(249, 179, 128), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
