import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.19, contrast=0.97, saturation=1.12, hue=0.02),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAutocontrast(p=0.1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
