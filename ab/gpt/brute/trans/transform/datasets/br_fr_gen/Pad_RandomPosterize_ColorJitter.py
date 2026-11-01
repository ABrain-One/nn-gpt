import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(103, 202, 144), padding_mode='reflect'),
    transforms.RandomPosterize(bits=5, p=0.43),
    transforms.ColorJitter(brightness=0.91, contrast=0.8, saturation=0.86, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
