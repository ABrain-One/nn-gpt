import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(92, 221, 190), padding_mode='edge'),
    transforms.RandomHorizontalFlip(p=0.19),
    transforms.ColorJitter(brightness=0.82, contrast=0.98, saturation=0.86, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
