import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.95, contrast=1.08, saturation=0.96, hue=0.05),
    transforms.Pad(padding=1, fill=(73, 192, 13), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.21, p=0.38),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
