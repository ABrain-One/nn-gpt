import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.17),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.82, contrast=1.1, saturation=0.91, hue=0.1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
