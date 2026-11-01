import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(118, 156, 11), padding_mode='reflect'),
    transforms.ColorJitter(brightness=1.11, contrast=0.97, saturation=0.83, hue=0.08),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
