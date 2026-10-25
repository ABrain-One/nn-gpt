import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.83, contrast=1.05, saturation=0.81, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
