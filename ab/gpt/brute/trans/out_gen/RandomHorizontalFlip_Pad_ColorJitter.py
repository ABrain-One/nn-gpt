import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.Pad(padding=1, fill=(255, 113, 82), padding_mode='reflect'),
    transforms.ColorJitter(brightness=1.2, contrast=0.92, saturation=0.93, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
