import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.05, contrast=0.85, saturation=0.95, hue=0.01),
    transforms.RandomGrayscale(p=0.32),
    transforms.CenterCrop(size=25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
