import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.97, contrast=0.98, saturation=0.98, hue=0.09),
    transforms.RandomRotation(degrees=26),
    transforms.RandomGrayscale(p=0.85),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
