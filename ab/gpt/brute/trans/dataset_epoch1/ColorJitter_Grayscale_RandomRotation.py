import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.07, contrast=1.15, saturation=0.89, hue=0.01),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(degrees=21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
