import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.08, contrast=0.83, saturation=1.04, hue=0.05),
    transforms.RandomGrayscale(p=0.71),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
