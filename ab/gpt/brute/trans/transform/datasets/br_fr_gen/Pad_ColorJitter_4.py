import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(71, 157, 184), padding_mode='reflect'),
    transforms.ColorJitter(brightness=1.19, contrast=1.06, saturation=0.8, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
