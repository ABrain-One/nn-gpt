import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(200, 11, 204), padding_mode='edge'),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=1.0, contrast=0.99, saturation=1.08, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
