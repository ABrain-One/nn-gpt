import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.91, contrast=0.84, saturation=1.1, hue=0.03),
    transforms.Pad(padding=3, fill=(70, 204, 19), padding_mode='reflect'),
    transforms.RandomInvert(p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
