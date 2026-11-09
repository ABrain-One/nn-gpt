import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.82, contrast=1.18, saturation=0.81, hue=0.04),
    transforms.RandomGrayscale(p=0.89),
    transforms.RandomRotation(degrees=25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
