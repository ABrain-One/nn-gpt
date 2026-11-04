import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.98, contrast=0.85, saturation=1.08, hue=0.07),
    transforms.RandomGrayscale(p=0.8),
    transforms.RandomVerticalFlip(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
