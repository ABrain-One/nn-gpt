import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.95, contrast=1.09, saturation=1.0, hue=0.07),
    transforms.Pad(padding=5, fill=(234, 89, 86), padding_mode='constant'),
    transforms.RandomPosterize(bits=7, p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
