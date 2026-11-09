import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.82),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=1.17, contrast=1.19, saturation=1.08, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
