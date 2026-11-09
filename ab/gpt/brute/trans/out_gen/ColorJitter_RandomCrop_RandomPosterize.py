import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.13, contrast=1.05, saturation=0.94, hue=0.04),
    transforms.RandomCrop(size=28),
    transforms.RandomPosterize(bits=6, p=0.27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
