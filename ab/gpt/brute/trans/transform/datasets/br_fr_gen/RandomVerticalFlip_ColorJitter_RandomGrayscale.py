import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.62),
    transforms.ColorJitter(brightness=1.18, contrast=0.99, saturation=1.15, hue=0.06),
    transforms.RandomGrayscale(p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
