import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(203, 115, 252), padding_mode='reflect'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAutocontrast(p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
