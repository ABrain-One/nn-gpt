import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(217, 14, 227), padding_mode='edge'),
    transforms.ColorJitter(brightness=1.08, contrast=1.11, saturation=0.98, hue=0.04),
    transforms.RandomAdjustSharpness(sharpness_factor=1.17, p=0.57),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
