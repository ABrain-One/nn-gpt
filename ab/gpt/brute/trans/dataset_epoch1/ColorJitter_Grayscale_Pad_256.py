import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.1, contrast=0.83, saturation=0.87, hue=0.02),
    transforms.Grayscale(num_output_channels=3),
    transforms.Pad(padding=0, fill=(228, 147, 241), padding_mode='edge'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
