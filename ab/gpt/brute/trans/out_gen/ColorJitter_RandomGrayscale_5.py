import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.01, contrast=1.19, saturation=0.85, hue=0.04),
    transforms.RandomGrayscale(p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
